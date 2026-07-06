from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = {
    "kw": ("executor", "reviewer"),
    "kr": ("executor", "reviewer"),
    "kernel_aw": ("executor", "reviewer"),
    "cache_ca": ("executor",),
    "attention_pa": ("executor",),
    "integrator_ia": ("executor",),
}


def test_workflow_nodes_use_owned_managed_autorun_adapters() -> None:
    for workflow, agent_nodes in WORKFLOWS.items():
        workflow_dir = ROOT / "workflow" / workflow
        adapter = workflow_dir / "nodes" / "managed_codex.py"
        executor = workflow_dir / "nodes" / "codex_executor" / "node.py"
        reviewer = workflow_dir / "nodes" / "reviewer" / "node.py"

        assert adapter.exists()
        adapter_text = adapter.read_text()
        assert "codex_goal_session.py" in adapter_text
        assert '"CODEX_AUTORUN_SANDBOX": "workspace-write"' in adapter_text
        assert '"gpt-5.6-sol"' in adapter_text

        nodes = {"executor": executor, "reviewer": reviewer}
        for node_name in agent_nodes:
            node = nodes[node_name]
            assert node.exists()
            text = node.read_text()
            assert "run_managed_codex" in text
            assert "codex exec" not in text
            assert "dangerously-bypass-approvals-and-sandbox" not in text
            assert "codex_exec_flags" not in text

        if "reviewer" not in agent_nodes:
            assert not reviewer.exists()


def test_workflow_cli_has_no_legacy_exec_injection_surface() -> None:
    for workflow in WORKFLOWS:
        text = (ROOT / "workflow" / workflow / "workflow.py").read_text()
        assert "--autorun-model" in text
        assert "--autorun-poll-sec" in text
        assert "--codex-exec-flags" not in text
        assert "CODEX_EXEC_FLAGS" not in text
        assert "dangerously-bypass-approvals-and-sandbox" not in text


if __name__ == "__main__":
    test_workflow_nodes_use_owned_managed_autorun_adapters()
    test_workflow_cli_has_no_legacy_exec_injection_surface()
    print("managed Codex workflow tests passed")
