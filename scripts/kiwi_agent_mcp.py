#!/usr/bin/env python3
"""MCP transport adapter for :mod:`kiwi_agent_gateway`.

The MCP dependency is intentionally isolated here so the gallery and its unit
tests can still be imported without installing the agent extras.
"""

from __future__ import annotations

import base64
import hmac
import json
import threading
import time
from typing import Annotated, Any, TypedDict

from kiwi_agent_gateway import AgentGatewayError


class RobotStatusOutput(TypedDict, total=False):
    ok: bool
    ready: bool
    reason: str
    session_id: str
    capture_count: int
    live: dict[str, Any]
    navigation: dict[str, Any]
    visual_search: dict[str, Any]
    safety: dict[str, Any]
    next_action: dict[str, Any]
    recovery: dict[str, Any] | None


class SearchOutput(TypedDict, total=False):
    ok: bool
    query: str
    session_id: str
    diversified: bool
    results: list[dict[str, Any]]


class MapOutput(TypedDict, total=False):
    ok: bool
    frame: str
    pose: dict[str, Any]
    localization_quality: dict[str, Any] | None
    include_path: bool
    map_bounds: dict[str, float]
    resolution_m: float
    keyframes: int


class PreviewOutput(TypedDict, total=False):
    ok: bool
    capture_ref: str
    goal_pose: dict[str, Any]
    straight_line_distance_m: float | None
    planned_path_distance_m: float | None
    max_travel_distance_m: float
    estimated_duration_s: float | None
    safe_to_start: bool
    blockers: list[str]
    route: list[dict[str, float]]
    preview_id: str
    expires_in_s: float
    expires_at: str


class NavigationOutput(TypedDict, total=False):
    ok: bool
    action_id: str | None
    phase: str
    capture_ref: str | None
    goal_pose: dict[str, Any]
    started_at: float | None
    finished_at: float | None
    distance_traveled_m: float
    planned_path_distance_m: float | None
    remaining_path_m: float | None
    cross_track_error_m: float | None
    stop_reason: str | None
    terminal: bool
    retry_after_s: float | None
    suggested_tool: str


class NavigationReportOutput(TypedDict, total=False):
    ok: bool
    action_id: str
    capture_ref: str
    phase: str
    duration_s: float
    pose_count: int
    camera_frame_count: int
    selected_frame_count: int
    distance_traveled_m: float
    measured_trace_distance_m: float
    planned_path_distance_m: float | None
    remaining_path_m: float | None
    cross_track_error_m: float | None
    stop_reason: str | None
    navigator_message: str | None
    logs: list[str]
    goal_pose: dict[str, Any]
    final_pose: dict[str, Any] | None
    camera_frames: list[dict[str, Any]]
    evidence_sources: list[str]
    simulator_ground_truth_used: bool


class StopOutput(TypedDict, total=False):
    ok: bool
    stopped: bool
    navigation: dict[str, Any]


def _mcp_result(result, types):
    structured = {"ok": True, **result.structured}
    content = [types.TextContent(
        type="text",
        text=json.dumps(structured, indent=2, sort_keys=True),
    )]
    for attachment in result.images:
        if attachment.path is not None:
            data = attachment.path.read_bytes()
        elif attachment.data is not None:
            data = attachment.data
        else:
            continue
        content.append(types.TextContent(type="text", text=attachment.label))
        content.append(types.ImageContent(
            type="image",
            data=base64.b64encode(data).decode("ascii"),
            mime_type=attachment.mime_type,
        ))
    return types.CallToolResult(
        content=content,
        structured_content=structured,
    )


def _call_gateway(call, types):
    try:
        return _mcp_result(call(), types)
    except AgentGatewayError as exc:
        structured = {"ok": False, "error": exc.as_dict()}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=str(exc))],
            structured_content=structured,
            is_error=True,
        )


def create_mcp_server(gateway):
    """Create the typed MCP surface around one gateway instance."""
    try:
        import mcp.types as types
        from mcp.server import MCPServer
    except ImportError as exc:
        raise RuntimeError(
            "MCP serving requires the agent dependencies; install "
            "requirements-agent.txt") from exc

    server = MCPServer(
        "kiwi-robot",
        title="Kiwi Robot Visual Navigation",
        description=(
            "Inspect and safely navigate a local Kiwi robot using visual "
            "place goals on its live SLAM map."),
        version="1.1.0",
        instructions=(
            "Before motion, call get_robot_status, search_goal_images or inspect "
            "the map, then preview_image_goal. navigate_to_image accepts only a "
            "fresh preview_id and moves physical hardware. Monitor it with "
            "get_navigation_status; stop_navigation is always safe to call. "
            "After or during an action, get_navigation_report returns camera "
            "and live-SLAM visual evidence without simulator ground truth. "
            "Similarity values rank images and are not calibrated confidence."),
    )
    read_only = types.ToolAnnotations(
        read_only_hint=True, destructive_hint=False,
        idempotent_hint=True, open_world_hint=False)

    def typed_tool(output_type, **kwargs):
        def decorator(function):
            function.__annotations__["return"] = Annotated[
                types.CallToolResult, output_type]
            return server.tool(structured_output=True, **kwargs)(function)
        return decorator

    @typed_tool(RobotStatusOutput, annotations=read_only)
    def get_robot_status():
        """Check pose/map freshness, relocalization, mux, session, and action state."""
        return _call_gateway(gateway.get_robot_status, types)

    @typed_tool(SearchOutput, annotations=read_only)
    def search_goal_images(query: str, top_n: int = 4,
                           diversify: bool = True):
        """Find visually relevant saved places and return ranked thumbnails."""
        return _call_gateway(
            lambda: gateway.search_goal_images(query, top_n, diversify), types)

    @typed_tool(MapOutput, annotations=read_only)
    def get_pose_on_map(view: str = "full", radius_m: float | None = None,
                        include_path: bool = True):
        """Render the live occupancy map with robot heading and active route."""
        return _call_gateway(
            lambda: gateway.get_pose_on_map(view, radius_m, include_path), types)

    @typed_tool(PreviewOutput, annotations=read_only)
    def preview_image_goal(capture_ref: str,
                           max_travel_distance_m: float):
        """Plan without moving and issue a short-lived preview when safe."""
        return _call_gateway(
            lambda: gateway.preview_image_goal(
                capture_ref, max_travel_distance_m),
            types,
        )

    @typed_tool(
        PreviewOutput,
        name="preview_navigation_to_image",
        annotations=read_only,
    )
    def preview_navigation_to_image(capture_ref: str,
                                    max_travel_distance_m: float):
        """Compatibility alias for preview_image_goal; plan without moving."""
        return _call_gateway(
            lambda: gateway.preview_image_goal(
                capture_ref, max_travel_distance_m),
            types,
        )

    @typed_tool(
        NavigationOutput,
        annotations=types.ToolAnnotations(
            read_only_hint=False, destructive_hint=True,
            idempotent_hint=False, open_world_hint=True),
    )
    def navigate_to_image(preview_id: str):
        """Revalidate a preview and start one budget-limited physical action."""
        return _call_gateway(
            lambda: gateway.navigate_to_image(preview_id), types)

    @typed_tool(NavigationOutput, annotations=read_only)
    def get_navigation_status(action_id: str | None = None,
                              wait_s: float = 0.0):
        """Read the active/latest action, optionally waiting briefly for change."""
        return _call_gateway(
            lambda: gateway.get_navigation_status(action_id, wait_s), types)

    @typed_tool(NavigationReportOutput, annotations=read_only)
    def get_navigation_report(action_id: str | None = None,
                              frame_count: int = 8,
                              brightness_gain: float = 1.0):
        """Return a camera contact sheet and measured SLAM trajectory for an action."""
        return _call_gateway(
            lambda: gateway.get_navigation_report(
                action_id, frame_count, brightness_gain),
            types,
        )

    @typed_tool(
        StopOutput,
        annotations=types.ToolAnnotations(
            read_only_hint=False, destructive_hint=False,
            idempotent_hint=True, open_world_hint=True),
    )
    def stop_navigation(action_id: str | None = None,
                        reason: str | None = None):
        """Idempotently stop agent navigation; this is not a latched emergency stop."""
        return _call_gateway(
            lambda: gateway.stop_navigation(action_id, reason), types)

    return server


class _BearerAuthApp:
    def __init__(self, app, token: str):
        self.app = app
        self.token = f"Bearer {token}".encode("utf-8")

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = {key.lower(): value for key, value in scope.get("headers", [])}
            authorization = headers.get(b"authorization", b"")
            if not hmac.compare_digest(authorization, self.token):
                body = b'{"error":"missing or invalid bearer token"}'
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                        (b"www-authenticate", b"Bearer"),
                    ],
                })
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


class McpServerHandle:
    """Run a shared-state MCP ASGI app alongside the gallery HTTP server."""

    def __init__(self, gateway, host: str, port: int,
                 bearer_token: str | None = None):
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "MCP serving requires uvicorn; install requirements-agent.txt") from exc
        mcp = create_mcp_server(gateway)
        app = mcp.streamable_http_app(
            streamable_http_path="/mcp",
            json_response=True,
            stateless_http=False,
            host=host,
        )
        if bearer_token:
            app = _BearerAuthApp(app, bearer_token)
        config = uvicorn.Config(
            app, host=host, port=int(port), log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            name="kiwi-agent-mcp",
            daemon=True,
        )

    def start(self, timeout_s: float = 5.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if getattr(self._server, "started", False):
                return
            if not self._thread.is_alive():
                raise RuntimeError("MCP server exited during startup")
            time.sleep(0.02)
        raise RuntimeError("timed out waiting for the MCP server to start")

    def stop(self, timeout_s: float = 5.0) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=max(0.0, timeout_s))

    @property
    def running(self) -> bool:
        return self._thread.is_alive()
