## Blind Video-Frame Quality Review

You are an independent visual evidence node. Judge only the attached comparison
images. Do not inspect repository code, workflow state, filenames outside the
attached list, implementation notes, or candidate identity. Do not run shell
commands. The optimizer and method name are intentionally hidden.

Each attached image is one aligned baseline/candidate frame pair. One source is
always on the left and the other is always on the right, with that assignment
held constant across all images. Images cover five prompts and four points per
prompt, including one adjacent-frame pair. Compare the two sides for newly introduced:

- subject or geometry corruption;
- motion-coherence evidence, flicker/popping, ghosting, or unstable detail;
- blur, texture/detail loss, static/noise, color or exposure shifts;
- patch/block boundaries, seams, duplicated content, or missing content;
- prompt-specific failures visible on only one side.

Judge differences, not baseline aesthetics. Use `low` only for a visible but
minor difference, `medium` for a clearly objectionable regression, `high` for a
major failure, and `critical` for unusable output. If no side is consistently
worse, use `neither`. If evidence is genuinely ambiguous, use `unclear`.
