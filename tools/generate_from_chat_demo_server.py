#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import logging
import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from llmparty.demo.chat.api_client import APIClient

try:
    from rich.logging import RichHandler
    from rich.progress import Progress
except ModuleNotFoundError:
    class RichHandler(logging.StreamHandler):
        def __init__(self, *args, **kwargs):
            super().__init__()

    Progress = None

try:
    from llmparty.dataset.dialog import LOSS_END, LOSS_START, CompressFormatDialog
except ModuleNotFoundError:
    LOSS_START = "<|loss_start|>"
    LOSS_END = "<|loss_end|>"
    CompressFormatDialog = None

try:
    from llmparty.scripts.command_base import CommandBase
    from llmparty.scripts.eval.compute_metrics import compute_metrics_from_file
    from llmparty.scripts.eval.merge_evaluated_files import merge_files
    from llmparty.scripts.eval.stats_metrics import record_metrics
    from llmparty.utils.misc import StreamJsonlSaver, load_dialogs
    from llmparty.utils.progress import tqdm_progress_format_for_rich
except ModuleNotFoundError:
    class CommandBase:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("llmparty CLI dependencies are unavailable in this lightweight app environment.")

    compute_metrics_from_file = None
    merge_files = None
    record_metrics = None
    StreamJsonlSaver = None
    load_dialogs = None

    def tqdm_progress_format_for_rich():
        return []

# logger
formatter = "%(message)s"
logging.basicConfig(
    level="DEBUG", format=formatter, datefmt="[%X]", handlers=[RichHandler(markup=True)]
)
logger = logging.getLogger("rich")

DEFAULT_URL = "http://localhost:9898"
DEFAULT_PROVIDER = "openai_api_like"


def generate(
    messages: list,
    model: str = None,
    tools: dict = None,
    provider: str = DEFAULT_PROVIDER,
    url: str = DEFAULT_URL,
    max_tokens: int = 64,
    temperature: float = 0.3,
    top_k: int = -1,
    top_p: float = 0.95,
    min_p: float = 0.0,
    top_n_sigma: float = 0.0,
    n_samples: int = 1,
    prefix: str = None,
    **kwargs,
):
    client = APIClient(
        model_name=model,
        provider=provider,
        url=url,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        top_n_sigma=top_n_sigma,
    )
    if tools:
        tools = list(tools.values())
    else:
        tools = None
    new_messages = copy.deepcopy(messages)
    for message in new_messages:
        if LOSS_START in message["content"]:
            message["content"] = message["content"].replace(LOSS_START, "")
        if LOSS_END in message["content"]:
            message["content"] = message["content"].replace(LOSS_END, "")
    response_messages = []
    for _ in range(n_samples):
        response_message = client.chat_completion(
            new_messages, tools=tools, prefix=prefix
        )
        response_messages.append(response_message)
    return response_messages


def generate_from_example_iter(iter_item: dict, **kwargs):
    turn_index = iter_item["turn_index"]
    messages = iter_item["messages"]
    tools = iter_item["tools"]
    if (
        "extras" in messages[-1]
        and isinstance(messages[-1]["extras"], dict)
        and "prefix" in messages[-1]["extras"]
    ):
        prefix = messages[-1]["extras"]["prefix"]
    else:
        prefix = None
    try:
        response_messages = generate(messages[:-1], tools=tools, prefix=prefix, **kwargs)
    except BaseException as e:
        raise ValueError(f"failed to generate for turn_index {turn_index}: {e}")

    # if kwargs.get("compute_metrics", False):
    #     # ! it is not work for parallel because nltk is not thread safe
    #     metrics = compute_metrics(
    #         messages[-1]["content"],
    #         response_message["content"],
    #         tools=tools,
    #     )
    # else:
    metrics = None
    return turn_index, response_messages, metrics


def generate_examples(
    example_tuple: tuple,
    model: str = None,
    serve_model_name: str = None,
    provider: str = DEFAULT_PROVIDER,
    url: str = DEFAULT_URL,
    max_tokens: int = 64,
    temperature: float = 0.3,
    top_k: int = -1,
    top_p: float = 0.95,
    min_p: float = 0.0,
    top_n_sigma: float = 0.0,
    turn_parallel: int = 1,
    progress: Progress = None,
    n_samples: int = 1,
    **kwargs,
):
    example_index, example = example_tuple
    cf_dialog = CompressFormatDialog(**example)
    pbar = progress.add_task(
        f"[red]{model}[/]-dialog-[yellow]{example_index+1}[/]",
        total=cf_dialog.get_expand_length(single_turn=True, skip_non_loss=True)
        * n_samples,
    )
    if not serve_model_name:
        serve_model_name = model
    try:
        with ThreadPoolExecutor(max_workers=turn_parallel) as executor:
            for turn_index, response_messages, metrics in executor.map(
                partial(
                    generate_from_example_iter,
                    model=serve_model_name,
                    provider=provider,
                    url=url,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    top_n_sigma=top_n_sigma,
                    n_samples=n_samples,
                ),
                cf_dialog.iter_from_expand(single_turn=True, skip_non_loss=True),
            ):
                if "evaluate" not in example["dialog"][turn_index] or kwargs.get(
                    "override", False
                ):
                    example["dialog"][turn_index]["evaluate"] = {}

                if len(response_messages) > 1:
                    for i, response_message in enumerate(response_messages):
                        example["dialog"][turn_index]["evaluate"][f"{model}-s{i}"] = dict(
                            content=response_message["content"],
                            metrics=metrics,
                            meta=dict(
                                provider=provider,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                min_p=min_p,
                                top_n_sigma=top_n_sigma,
                            ),
                        )
                else:
                    example["dialog"][turn_index]["evaluate"] = {
                        model: dict(
                            content=response_messages[0]["content"],
                            metrics=metrics,
                            meta=dict(
                                provider=provider,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                min_p=min_p,
                                top_n_sigma=top_n_sigma,
                            ),
                        )
                    }
                progress.update(task_id=pbar, advance=n_samples)
    except BaseException as e:
        raise ValueError(f"Failed to genrate for example_index {example_index}: {e}")
    progress.update(task_id=pbar, visible=False)
    return example


def generate_model_parallel(
    model_tuple: tuple,
    examples: list = None,
    example_parallel: int = 1,
    progress: Progress = None,
    model_maps: dict = None,
    **kwargs,
):
    model_index, provider, model, output_path = model_tuple
    logger.info(
        f"Run model: {model_index+1} [blue]{provider}:{model}[/] -> [green]{output_path}[/]"
    )
    saver = StreamJsonlSaver(output_path)
    pbar = progress.add_task(f"[green]{model}[/]", total=len(examples))
    examples = copy.deepcopy(examples)
    success = True
    try:
        with ThreadPoolExecutor(max_workers=example_parallel) as executor:
            for example in executor.map(
                partial(
                    generate_examples,
                    model=model,
                    serve_model_name=model_maps[model]
                    if model_maps and model in model_maps
                    else None,
                    provider=provider,
                    progress=progress,
                    **kwargs,
                ),
                enumerate(examples),
            ):
                progress.update(task_id=pbar, advance=1)
                saver.save(example)
    except BaseException as e:
        logger.error(f"Error for [red]{provider}:{model}[/]: {e}", exc_info=True)
        success = False
    finally:
        progress.update(task_id=pbar, visible=False)
        saver.close()
    return success


def clean_evaluate(examples: list):
    for example in examples:
        for turn in example["dialog"]:
            if "evaluate" in turn:
                del turn["evaluate"]
    return examples

class GenerateFromChatDemoServer(CommandBase):
    @classmethod
    def add_arguments(cls, parser=None):
        parser = super().add_arguments(parser)
        parser.add_argument(
            "--force", "-f", action="store_true", help="overwrite exist file"
        )
        parser.add_argument("--start", "-s", type=int, default=0, help="start index")
        parser.add_argument("--end", "-e", type=int, default=0, help="end index")
        parser.add_argument(
            "--url", "-u", type=str, default=DEFAULT_URL, help="server url"
        )
        parser.add_argument(
            "--max-tokens", "-m", type=int, default=64, help="max tokens"
        )
        parser.add_argument(
            "--temperature", "-t", type=float, default=0.3, help="temperature"
        )
        parser.add_argument("--top-k", "-k", type=int, default=-1, help="top k")
        parser.add_argument("--top-p", "-p", type=float, default=0.95, help="top p")
        parser.add_argument("--min-p", type=float, default=0.0, help="min p")
        parser.add_argument(
            "--top-n-sigma", "-ns", type=float, default=0.0, help="top n sigma"
        )
        parser.add_argument("--n-samples", "-n", type=int, default=1, help="n samples")
        parser.add_argument(
            "--model-parallel", "-mp", type=int, default=-1, help="model parallel"
        )
        parser.add_argument(
            "--example-parallel", "-ep", type=int, default=1, help="example parallel"
        )
        parser.add_argument(
            "--turn-parallel", "-tp", type=int, default=1, help="turn parallel"
        )
        parser.add_argument(
            "--compute-metrics", "-cm", action="store_true", help="compute metrics"
        )
        parser.add_argument(
            "--show-level", "-sl", action="store_true", help="show level-specific stats"
        )
        parser.add_argument(
            "--models",
            type=str,
            nargs="+",
            required=True,
            help="models, [provider:]model[--serve_name]",
        )
        # Main
        parser.add_argument(
            "--input",
            "-i",
            type=str,
            help="a compress format json of input file",
            required=True,
        )

        parser.add_argument(
            "--output",
            "-o",
            type=str,
            help="a compress format json of output file",
            required=True,
        )
        parser.add_argument(
            "--override",
            "-ov",
            action="store_true",
            help="if true, override the exist results",
        )
        parser.add_argument(
            "--roles",
            "-r",
            type=str,
            nargs="+",
            default=["assistant"],
            help="The metrics of target role to stats",
        )
        parser.add_argument(
            "--output-example-details",
            "-oe",
            type=str,
            nargs="+",
            help="a file to save details of each example",
        )
        parser.add_argument(
            "--output-stats-table",
            "-ot",
            type=str,
            nargs="+",
            help="files to save final table of stats",
        )
        parser.add_argument(
            "--output-dashboard",
            "-ob",
            type=str,
            help="The path of jsonl to output dashboard",
        )

        return parser


    def run(self):
        # ------- Write your main code -------#
        if not self.args.output.endswith(".jsonl"):
            raise ValueError("Expected a .jsonl output file.")
        logger.info("Load dialogs from [green]{}[/]".format(self.args.input))
        raw_examples = load_dialogs(self.args.input)
        start = self.args.start
        if self.args.end == 0:
            end = len(raw_examples)
        else:
            end = self.args.end

        raw_examples = clean_evaluate(raw_examples[start:end])
        if len(raw_examples) == 0:
            raise ValueError(
                f"No examples will be generated with start={0}, end={end}"
            )
        logger.info(f"{len(raw_examples)} examples are selected.")
        logger.info("Check models")
        all_output_paths = []
        model_tuples = []
        model_index = 0
        model_set = set()
        model_maps = dict()
        for model in self.args.models:
            if ":" in model:
                provider, model = model.split(":", 1)
            else:
                provider = DEFAULT_PROVIDER

            if "--" in model:
                model, call_name = model.split("--", 1)
                model_maps[model] = call_name

            output_path = self.args.output.replace(".jsonl", f"/{model}.jsonl")
            all_output_paths.append(output_path)
            if pathlib.Path(output_path).exists() and not self.args.force:
                logger.warning(
                    f"> Skip model with exist file: {model} -> [red]{output_path}[/]"
                )
                continue
            if model in model_set:
                raise ValueError(f"Duplicate model: {model}")
            model_set.add(model)
            model_tuples.append((model_index, provider, model, output_path))
            model_index += 1

        # run model
        model_parallel = min(self.args.model_parallel, len(model_tuples))
        if model_parallel <= 0:
            model_parallel = len(model_tuples)

        logger.info(f"Run {len(model_tuples)} models with {model_parallel} jobs")
        if len(model_tuples) > 0:
            status = []
            with Progress(*tqdm_progress_format_for_rich()) as progress:
                pbar = progress.add_task(
                    "[blue]Running models[/]", total=len(model_tuples)
                )
                with ThreadPoolExecutor(max_workers=model_parallel) as executor:
                    for x in executor.map(
                        partial(
                            generate_model_parallel,
                            examples=raw_examples,
                            progress=progress,
                            model_maps=model_maps,
                            **vars(self.args),
                        ),
                        model_tuples,
                    ):
                        progress.update(task_id=pbar, advance=1)
                        status.append(x)

            if sum(status) != len(model_tuples):
                raise RuntimeError(
                    f"{len(model_tuples)-sum(status)}/{len(model_tuples)} model tasks failed."
                )

        if self.args.compute_metrics:
            logger.info("Compute metrics")
            for sub_path in all_output_paths:
                compute_metrics_from_file(sub_path, sub_path)

        if len(all_output_paths) == 1:
            # copy file
            logger.info(
                f"Copy file from [green]{all_output_paths[0]}[/] to [green]{self.args.output}[/]"
            )
            shutil.copyfile(all_output_paths[0], self.args.output)
        else:
            # merge all models and rm temp files
            logger.info(
                f"Merge {len(all_output_paths)} models results -> [green]{self.args.output}[/]"
            )
            merge_files(all_output_paths, self.args.output)

        if self.args.compute_metrics:
            logger.info(f"Stats metrics for [green]{self.args.output}[/]")
            examples = load_dialogs(self.args.output)
            record_metrics(
                examples,
                show_level=self.args.show_level,
                roles=self.args.roles,
                eval_file=self.args.output,
                output_stats_table=self.args.output_stats_table,
                output_example_details=self.args.output_example_details,
                output_dashboard=self.args.output_dashboard,
            )
        logger.info("All Done.")
        # ------------------------------------#


if __name__ == "__main__":
    GenerateFromChatDemoServer()
