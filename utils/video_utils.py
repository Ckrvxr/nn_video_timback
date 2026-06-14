def variant_dirname(encoder: str, crf: int, preset: int) -> str:
    return f'{encoder}_crf{crf}_p{preset}'
