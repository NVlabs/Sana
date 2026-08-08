#!/usr/bin/env python3
"""Run the aligned MiniMax-H3 RTX 5090 review prompt suite."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ["H3_SUITE_OUTPUT_ROOT"])
SERVER_URL = os.environ["H3_SERVER_URL"]
NUM_STEPS = int(os.getenv("H3_NUM_STEPS", "50"))
DURATION_SECONDS = int(os.getenv("H3_DURATION_SECONDS", "5"))

CASES = [
    {
        "slug": "flamenco_dancer",
        "seed": 1301,
        "prompt": (
            "A flamenco dancer spins fast on a wooden stage under a single warm "
            "spotlight, her red skirt flaring wide, heels striking the boards in "
            "sharp rapid bursts, a guitar driving underneath and hands clapping on "
            "the offbeat, camera circling with her."
        ),
    },
    {
        "slug": "garage_mechanics",
        "seed": 1302,
        "prompt": (
            "Two mechanics in a garage at night lean over an open engine bay. The "
            "older one wipes his hands and says \"It's not the pump.\" The younger "
            "one answers \"Then what is it?\" A radio murmurs in the corner, a "
            "wrench clatters onto the concrete, fluorescent tubes buzz overhead."
        ),
    },
    {
        "slug": "official_starship",
        "seed": 0,
        "prompt": """integrated_multimodal_description: [Shot 1] Cinematic, medium wide shot, pushing in slowly. In the cavernous, dimly lit bridge of a starship, sleek metallic consoles with glowing amber displays flank a massive, curved observation window. A female captain, in her late 40s with an athletic build and short silver-streaked black hair, stands in the center midground. She wears a structured, high-collared dark navy military tunic with silver chest insignias. Her back is to the camera, silhouetted against the cool, ambient starlight pouring through the thick glass. She stands perfectly still with her hands clasped tightly behind her back. Outside the window, a massive armada of jagged, dark grey dreadnoughts hovers in tight formation against a deep purple space nebula. The fleet's massive rear thrusters begin to glow with an intense, escalating bright blue light. [Shot 2] At 00:04.500, the camera cuts to a close-up of the captain's face and shakes strongly. The brilliant blue-white light from the fleet's gathering energy reflects vividly in her dark eyes. Suddenly, a blinding white flash floods through the window, completely washing out the background as the fleet jumps to hyperspace. The sheer spatial force violently jolts the bridge, causing the captain from Shot 1 to stagger slightly forward, her shoulders tensing as she visibly braces herself against the physical tremors. As the intense white light fades abruptly, leaving only the dim, empty expanse of the purple nebula reflected on her starkly lit skin, her jaw clenches, and she slowly closes her eyes in the newly emptied space.
overall_soundscape: A low, resonant hum of the ship's ambient life support systems serves as the baseline, soon drowned out by an audible, escalating, high-pitched electronic whine as the fleet outside charges its hyperdrives. A massive, deafening, bass-heavy boom and sharp crackle erupts during the blinding flash, accompanied by the loud metallic creaking, rattling, and deep thuds of the bridge's bulkheads vibrating under immense physical stress. The intense roaring impact then cuts abruptly back to a hollow, echoing room tone, leaving only the faint, steady hum of the isolated bridge.
non_diegetic_music: Cinematic space-opera orchestral score, slow tempo, featuring a solitary, mournful French horn melody over deep, sustained string dissonances that build rapidly in volume and intensity, swelling to a massive orchestral peak before snapping immediately into silence right after the jump.""",
    },
]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "server_url": SERVER_URL,
        "num_steps": NUM_STEPS,
        "duration_seconds": DURATION_SECONDS,
        "cases": CASES,
    }
    (OUTPUT_ROOT / "suite_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )

    for case in CASES:
        case_dir = OUTPUT_ROOT / case["slug"]
        env = os.environ.copy()
        env.update(
            {
                "H3_SERVER_URL": SERVER_URL,
                "H3_NUM_STEPS": str(NUM_STEPS),
                "H3_DURATION_SECONDS": str(DURATION_SECONDS),
                "H3_SEED": str(case["seed"]),
                "H3_PROMPT": case["prompt"],
                "H3_OUTPUT_DIR": str(case_dir),
            }
        )
        with (OUTPUT_ROOT / f"{case['slug']}.log").open("w") as log:
            subprocess.run(
                [
                    os.getenv("PYTHON_BIN", "python3"),
                    str(RUNTIME_ROOT / "request.py"),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


if __name__ == "__main__":
    main()
