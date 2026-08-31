import json
import pathlib
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_spatial_memory_timeline import (  # noqa: E402
    load_timeline,
    render_html,
    resolve_rollout,
)


class SpatialMemoryTimelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.thread = "01a040eb-b34d-7db3-afd5-5cf8545994d8"
        self.rollout = self.root / f"rollout-2026-08-26-{self.thread}.jsonl"
        records = [
            {
                "timestamp": "2026-08-27T01:53:30Z",
                "type": "session_meta",
                "payload": {"id": self.thread, "title": "Robot errand"},
            },
            {
                "timestamp": "2026-08-27T01:53:32Z",
                "type": "event_msg",
                "payload": {"type": "item_completed", "item": {
                    "type": "McpToolCall", "id": "one", "server": "kiwi",
                    "tool": "search_goal_images", "status": "completed",
                    "arguments": {"query": "hamper"},
                    "result": {"structuredContent": {"ok": True, "results": [
                        {"rank": 1, "capture_ref": "s:1",
                         "pose": {"x": 1.0, "y": 2.0}}
                    ]}},
                }},
            },
            {
                "timestamp": "2026-08-27T01:53:35Z",
                "type": "event_msg",
                "payload": {"type": "item_completed", "item": {
                    "type": "McpToolCall", "id": "two", "server": "kiwi",
                    "tool": "get_navigation_report", "status": "completed",
                    "arguments": {"action_id": "a"},
                    "result": {"structuredContent": {
                        "ok": True, "action_id": "a", "phase": "succeeded",
                        "distance_traveled_m": 1.25,
                    }},
                }},
            },
            {
                "timestamp": "2026-08-27T01:53:36Z",
                "type": "event_msg",
                "payload": {"type": "item_completed", "item": {
                    "type": "McpToolCall", "server": "other",
                    "tool": "search_goal_images", "status": "completed",
                }},
            },
        ]
        self.rollout.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_resolves_thread_urls(self):
        resolved = resolve_rollout(f"codex://threads/{self.thread}", self.root)
        self.assertEqual(resolved, self.rollout.resolve())

    def test_extracts_only_kiwi_spatial_calls(self):
        data = load_timeline(self.rollout, include_images=False)
        self.assertEqual(data["event_count"], 2)
        self.assertEqual(data["navigation_count"], 1)
        self.assertEqual(data["distance_m"], 1.25)
        self.assertEqual(data["events"][1]["elapsed_s"], 3.0)
        self.assertEqual(data["events"][0]["lane"], "memory")

    def test_renders_self_contained_html(self):
        page = render_html(load_timeline(self.rollout, include_images=False))
        self.assertIn("<!doctype html>", page)
        self.assertIn("Robot errand", page)
        self.assertIn("search_goal_images", page)
        self.assertNotIn("https://", page)


if __name__ == "__main__":
    unittest.main()
