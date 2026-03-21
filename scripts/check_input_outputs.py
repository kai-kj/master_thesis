from pathlib import Path


def check_input_outputs(
        inputs: dict[str, Path] | None,
        outputs: dict[str, Path] | None,
):
    if inputs is not None:
        for k, v in inputs.items():
            assert v.exists(), f"{k} ({v.as_posix()}) does not exist"

    if outputs is not None:
        for _, v in outputs.items():
            v.parent.mkdir(parents=True, exist_ok=True)
