from pathlib import Path


def check_input_outputs(
        inputs: dict[str, Path | None] | None,
        outputs: dict[str, Path | None] | None,
):
    if inputs is not None:
        for k, v in inputs.items():
            if v is not None:
                assert v.exists(), f"{k} ({v.as_posix()}) does not exist"

    if outputs is not None:
        for _, v in outputs.items():
            if v is not None:
                v.parent.mkdir(parents=True, exist_ok=True)
