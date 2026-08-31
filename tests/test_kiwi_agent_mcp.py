import asyncio
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

try:
    import mcp  # noqa: F401
except ImportError:
    mcp = None

from kiwi_agent_gateway import (  # noqa: E402
    AgentGatewayError,
    GatewayResult,
    ImageAttachment,
)
from kiwi_agent_mcp import create_mcp_server  # noqa: E402


class FakeGateway:
    def get_robot_status(self):
        return GatewayResult(
            {"ready": True, "reason": "ready to drive"},
            (ImageAttachment("map", "image/png", data=b"png"),),
        )

    def navigate_to_image(self, preview_id):
        raise AgentGatewayError(
            "preview_id has expired", code="preview_expired",
            retryable=True, suggested_tool="preview_image_goal")


@unittest.skipIf(mcp is None, "MCP agent dependencies are not installed")
class McpSurfaceTests(unittest.TestCase):
    def test_registers_typed_tool_surface_and_motion_annotations(self):
        server = create_mcp_server(FakeGateway())

        tools = asyncio.run(server.list_tools())
        by_name = {tool.name: tool for tool in tools}

        self.assertEqual(set(by_name), {
            "get_robot_status",
            "search_goal_images",
            "get_pose_on_map",
            "preview_image_goal",
            "preview_navigation_to_image",
            "navigate_to_image",
            "get_navigation_status",
            "get_navigation_report",
            "stop_navigation",
        })
        motion = by_name["navigate_to_image"].annotations
        self.assertFalse(motion.read_only_hint)
        self.assertTrue(motion.destructive_hint)
        self.assertTrue(motion.open_world_hint)
        stop = by_name["stop_navigation"].annotations
        self.assertTrue(stop.idempotent_hint)
        preview_schema = by_name["preview_image_goal"].output_schema
        self.assertIn("safe_to_start", preview_schema["properties"])
        self.assertIn("planned_path_distance_m", preview_schema["properties"])
        self.assertIn("preview_id", preview_schema["properties"])
        report_schema = by_name["get_navigation_report"].output_schema
        self.assertIn("simulator_ground_truth_used", report_schema["properties"])
        self.assertIn("navigator_message", report_schema["properties"])
        self.assertIn("logs", report_schema["properties"])

    def test_returns_structured_content_and_image_blocks(self):
        server = create_mcp_server(FakeGateway())

        result = asyncio.run(server.call_tool("get_robot_status", {}))

        self.assertEqual(result.structured_content["ready"], True)
        self.assertEqual([item.type for item in result.content],
                         ["text", "text", "image"])
        self.assertEqual(result.content[-1].mime_type, "image/png")

    def test_safe_gateway_failures_are_returned_as_tool_errors(self):
        server = create_mcp_server(FakeGateway())

        result = asyncio.run(server.call_tool(
            "navigate_to_image", {"preview_id": "expired"}))

        self.assertTrue(result.is_error)
        self.assertEqual(
            result.structured_content,
            {
                "ok": False,
                "error": {
                    "code": "preview_expired",
                    "message": "preview_id has expired",
                    "retryable": True,
                    "suggested_tool": "preview_image_goal",
                    "details": {},
                },
            })


if __name__ == "__main__":
    unittest.main()
