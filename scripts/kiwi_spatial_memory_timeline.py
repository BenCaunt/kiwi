#!/usr/bin/env python3
"""Build a self-contained spatial-memory timeline from a Codex rollout.

The input can be a rollout JSONL path, a Codex thread UUID, or a
``codex://threads/<uuid>`` URL.  Thread IDs are resolved below Codex's local
session directory.  No network access is required.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import pathlib
import re
import sys
from typing import Any


SPATIAL_TOOLS = {
    "get_robot_status": "state",
    "search_goal_images": "memory",
    "get_pose_on_map": "state",
    "preview_image_goal": "planning",
    "preview_navigation_to_image": "planning",
    "navigate_to_image": "movement",
    "get_navigation_status": "movement",
    "get_navigation_report": "evidence",
    "stop_navigation": "movement",
}


def resolve_rollout(source: str, sessions_dir: pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(source).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    match = re.search(
        r"(?:codex://threads/)?([0-9a-f]{8}-[0-9a-f-]{27,})", source,
        re.IGNORECASE,
    )
    if not match:
        raise FileNotFoundError(f"not a rollout path or Codex thread ID: {source}")
    thread_id = match.group(1)
    hits = sorted(sessions_dir.rglob(f"*{thread_id}.jsonl"))
    if not hits:
        raise FileNotFoundError(
            f"no local rollout found for {thread_id} below {sessions_dir}"
        )
    return hits[-1].resolve()


def parse_timestamp(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _first_image(result: dict[str, Any]) -> str | None:
    for block in result.get("content", []):
        if block.get("type") == "image" and block.get("data"):
            mime = block.get("mimeType") or block.get("mime_type") or "image/jpeg"
            return f"data:{mime};base64,{block['data']}"
    return None


def _compact_result(tool: str, data: dict[str, Any]) -> dict[str, Any]:
    """Keep useful spatial fields while avoiding giant logs and image payloads."""
    common = (
        "ok", "ready", "reason", "query", "session_id", "diversified",
        "capture_ref", "phase", "action_id", "safe_to_start", "blockers",
        "straight_line_distance_m", "planned_path_distance_m",
        "estimated_duration_s", "max_travel_distance_m",
        "distance_traveled_m", "remaining_path_m", "cross_track_error_m",
        "navigator_status", "navigator_message", "stop_reason", "terminal",
        "duration_s", "pose_count", "camera_frame_count",
        "selected_frame_count", "goal_pose", "goal", "final_pose", "pose",
        "route", "results", "error", "message", "started_at", "finished_at",
    )
    compact = {key: data[key] for key in common if key in data}
    if tool == "get_robot_status" and isinstance(data.get("live"), dict):
        live = data["live"]
        compact["live"] = {
            key: live[key] for key in
            ("status_code", "pose", "pose_age_s", "map_age_s") if key in live
        }
    if tool == "get_pose_on_map":
        for key in ("view", "localization_quality", "map_bounds", "keyframes"):
            if key in data:
                compact[key] = data[key]
    return compact


def load_timeline(path: pathlib.Path, include_images: bool = True) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
            payload = record.get("payload", {})
            if record.get("type") == "session_meta" and isinstance(payload, dict):
                metadata = payload
            if record.get("type") != "event_msg" or not isinstance(payload, dict):
                continue
            if payload.get("type") != "item_completed":
                continue
            item = payload.get("item", {})
            if item.get("type") != "McpToolCall" or item.get("server") != "kiwi":
                continue
            tool = item.get("tool")
            if tool not in SPATIAL_TOOLS:
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            structured = result.get("structuredContent")
            if not isinstance(structured, dict):
                structured = {}
            timestamp = parse_timestamp(record["timestamp"])
            event = {
                "id": item.get("id", f"event-{len(events) + 1}"),
                "timestamp": timestamp.isoformat(),
                "tool": tool,
                "lane": SPATIAL_TOOLS[tool],
                "status": item.get("status", "unknown"),
                "arguments": item.get("arguments", {}),
                "result": _compact_result(tool, structured),
            }
            if include_images and tool in {
                "search_goal_images", "get_pose_on_map", "get_navigation_report"
            }:
                event["image"] = _first_image(result)
            events.append(event)

    if not events:
        raise ValueError(f"no Kiwi spatial MCP calls found in {path}")
    events.sort(key=lambda event: event["timestamp"])
    start = parse_timestamp(events[0]["timestamp"])
    end = parse_timestamp(events[-1]["timestamp"])
    for index, event in enumerate(events):
        stamp = parse_timestamp(event["timestamp"])
        event["index"] = index + 1
        event["elapsed_s"] = round((stamp - start).total_seconds(), 3)

    reports: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["tool"] == "get_navigation_report":
            action_id = event["result"].get("action_id")
            if action_id:
                reports[action_id] = event["result"]
    total_distance = sum(
        float(report.get("distance_traveled_m") or 0) for report in reports.values()
    )
    return {
        "title": metadata.get("title") or "Kiwi spatial-memory run",
        "thread_id": metadata.get("id") or path.stem.rsplit("-", 1)[-1],
        "rollout": str(path),
        "started_at": start.isoformat(),
        "duration_s": round((end - start).total_seconds(), 3),
        "event_count": len(events),
        "navigation_count": len(reports),
        "distance_m": round(total_distance, 3),
        "events": events,
    }


def render_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(data["title"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — spatial memory timeline</title>
<style>
:root{{--ink:#16221b;--muted:#667168;--paper:#f4f1e9;--panel:#fffdf8;--line:#d7d4ca;--memory:#cd5b32;--planning:#c3982f;--movement:#26746c;--evidence:#5872aa;--state:#7763a6;--bad:#a53d3d}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
header{{padding:28px clamp(18px,4vw,58px) 22px;border-bottom:1px solid var(--line);background:linear-gradient(120deg,#fffdf8,#edf3e9)}}
h1{{font:700 clamp(25px,4vw,42px)/1.08 ui-serif,Georgia,serif;margin:0 0 8px;letter-spacing:-.03em}} .sub{{color:var(--muted);overflow-wrap:anywhere}}
.metrics{{display:flex;gap:28px;flex-wrap:wrap;margin-top:22px}} .metric b{{display:block;font:700 22px/1 ui-serif,Georgia,serif}} .metric span{{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}}
main{{max-width:1500px;margin:auto;padding:24px clamp(14px,3vw,38px) 50px}} .section-title{{font:700 19px ui-serif,Georgia,serif;margin:0 0 12px}}
.top{{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(310px,.75fr);gap:18px;align-items:stretch}}
.panel{{background:var(--panel);border:1px solid var(--line)}} .panel-head{{padding:12px 15px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px;align-items:center}}
.legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--muted)}} .sw{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;background:var(--c)}}
#map{{width:100%;height:390px;display:block;background-image:linear-gradient(#ece9df 1px,transparent 1px),linear-gradient(90deg,#ece9df 1px,transparent 1px);background-size:28px 28px}}
#map text{{font-size:11px;fill:var(--muted)}} .route{{fill:none;stroke:var(--planning);stroke-width:2.2;opacity:.66}} .route.sel{{stroke-width:4;opacity:1}} .candidate{{fill:var(--memory);opacity:.42}} .candidate.sel{{opacity:1;stroke:var(--ink);stroke-width:2}} .goal{{fill:var(--movement);stroke:var(--panel);stroke-width:2}}
.detail{{padding:16px;min-height:390px;overflow:auto}} .detail h3{{font:700 20px ui-serif,Georgia,serif;margin:0 0 5px}} .eyebrow{{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}} .detail img{{width:100%;max-height:200px;object-fit:contain;background:#e6e3da;margin:14px 0;border:1px solid var(--line)}}
.facts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:14px 0}} .fact{{border-top:1px solid var(--line);padding-top:7px;overflow-wrap:anywhere}} .fact span{{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}}
details{{margin-top:12px}} summary{{cursor:pointer;color:var(--muted)}} pre{{font:11px/1.45 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;background:#f1efe8;padding:10px;max-height:240px;overflow:auto}}
.timeline-panel{{margin-top:18px;overflow:hidden}} #timeline{{position:relative;padding:32px 16px 12px 104px;min-height:278px;overflow-x:auto}} .axis{{position:absolute;left:104px;right:16px;top:14px;height:18px;border-top:1px solid var(--line)}} .tick{{position:absolute;top:-5px;border-left:1px solid var(--line);height:258px;color:var(--muted);font-size:10px;padding:6px 0 0 5px;pointer-events:none}}
.lane{{height:52px;position:relative;border-top:1px solid #e8e5dd;min-width:650px}} .lane-label{{position:absolute;right:calc(100% + 14px);top:17px;width:84px;text-align:right;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-size:10px}}
.event{{position:absolute;top:13px;transform:translateX(-50%);width:24px;height:24px;border-radius:50%;border:3px solid var(--panel);background:var(--c);box-shadow:0 0 0 1px color-mix(in srgb,var(--c),#000 20%);cursor:pointer;color:white;font:700 9px/18px monospace;padding:0;z-index:2}} .event:hover,.event.sel{{transform:translateX(-50%) scale(1.3);z-index:5;outline:2px solid var(--ink);outline-offset:2px}} .event.failed{{background:var(--bad)}}
.navbar{{position:absolute;top:17px;height:16px;border-radius:2px;background:color-mix(in srgb,var(--movement),transparent 38%);min-width:3px;pointer-events:none}}
.stream{{margin-top:18px}} .stream-tools{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}} .filter{{background:transparent;border:1px solid var(--line);padding:6px 9px;cursor:pointer;font:inherit;color:var(--muted)}} .filter[aria-pressed=true]{{border-color:var(--ink);color:var(--ink);background:var(--panel)}}
.event-list{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:8px}} .event-card{{text-align:left;background:var(--panel);border:1px solid var(--line);padding:12px;cursor:pointer;color:inherit;font:inherit;min-height:100px}} .event-card:hover,.event-card.sel{{border-color:var(--ink)}} .event-card .num{{color:var(--muted);font-size:10px}} .event-card strong{{display:block;margin:5px 0;color:var(--c)}} .event-card p{{margin:0;color:var(--muted);font-size:12px}}
@media(max-width:820px){{.top{{grid-template-columns:1fr}} #map{{height:320px}} .detail{{min-height:0}}}}
</style></head><body>
<header><h1 id="title"></h1><div class="sub" id="subtitle"></div><div class="metrics" id="metrics"></div></header>
<main><div class="top"><section class="panel"><div class="panel-head"><h2 class="section-title">Spatial trace</h2><div class="legend"><span><i class="sw" style="--c:var(--memory)"></i>memory hit</span><span><i class="sw" style="--c:var(--planning)"></i>planned route</span><span><i class="sw" style="--c:var(--movement)"></i>executed goal</span></div></div><svg id="map" role="img" aria-label="Map-relative memory hits, planned routes, and navigation goals"></svg></section>
<aside class="panel detail" id="detail" aria-live="polite"></aside></div>
<section class="panel timeline-panel"><div class="panel-head"><h2 class="section-title">Call timeline</h2><span class="sub">select any event to inspect it</span></div><div id="timeline"></div></section>
<section class="stream"><div class="stream-tools" id="filters"></div><div class="event-list" id="event-list"></div></section></main>
<script id="run-data" type="application/json">{payload}</script><script>
const run=JSON.parse(document.getElementById('run-data').textContent); const colors={{memory:'var(--memory)',planning:'var(--planning)',movement:'var(--movement)',evidence:'var(--evidence)',state:'var(--state)'}}; let selected=0; let active=new Set(Object.keys(colors));
const fmt=n=>Number.isFinite(+n)?(+n).toFixed(2):'—'; const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
document.getElementById('title').textContent=run.title; document.getElementById('subtitle').textContent=`${{run.thread_id}} · ${{run.rollout}}`;
document.getElementById('metrics').innerHTML=[[run.event_count,'spatial calls'],[run.duration_s.toFixed(1)+' s','observed span'],[run.navigation_count,'navigation runs'],[run.distance_m.toFixed(2)+' m','distance traveled']].map(x=>`<div class="metric"><b>${{x[0]}}</b><span>${{x[1]}}</span></div>`).join('');
function xy(p){{if(!p||!Number.isFinite(+p.x)||!Number.isFinite(+p.y))return null;return {{x:+p.x,y:+p.y}}}}
function spatial(){{const pts=[],routes=[],goals=[];run.events.forEach((e,i)=>{{const r=e.result||{{}};(r.results||[]).forEach(v=>{{const p=xy(v.pose);if(p){{p.event=i;p.rank=v.rank;p.ref=v.capture_ref;pts.push(p)}}}});if(Array.isArray(r.route)){{const route=r.route.map(xy).filter(Boolean);route.event=i;if(route.length)routes.push(route)}};const g=xy(r.goal_pose||r.goal);if(g&&e.tool==='navigate_to_image'){{g.event=i;g.ref=r.capture_ref;goals.push(g)}}}});return {{pts,routes,goals}}}}
function drawMap(){{const svg=document.getElementById('map'),S=spatial(),all=[...S.pts,...S.goals,...S.routes.flat()];svg.innerHTML='';if(!all.length){{svg.innerHTML='<text x="20" y="32">No map-relative positions in this run.</text>';return}}const W=svg.clientWidth||700,H=svg.clientHeight||390,pad=34;const xs=all.map(p=>p.x),ys=all.map(p=>p.y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),dx=Math.max(.5,xmax-xmin),dy=Math.max(.5,ymax-ymin),scale=Math.min((W-pad*2)/dx,(H-pad*2)/dy);const X=x=>pad+(x-xmin)*scale+(W-pad*2-dx*scale)/2,Y=y=>H-pad-(y-ymin)*scale-(H-pad*2-dy*scale)/2;const ns='http://www.w3.org/2000/svg';const mk=(tag,attrs)=>{{const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));svg.appendChild(n);return n}};S.routes.forEach((r,j)=>mk('polyline',{{points:r.map(p=>`${{X(p.x)}},${{Y(p.y)}}`).join(' '),class:'route '+(r.event===selected?'sel':''),'data-event':r.event}}));S.pts.forEach(p=>{{const c=mk('circle',{{cx:X(p.x),cy:Y(p.y),r:p.event===selected?7:4,class:'candidate '+(p.event===selected?'sel':'')}});c.addEventListener('click',()=>select(p.event))}});S.goals.forEach(p=>{{const c=mk('circle',{{cx:X(p.x),cy:Y(p.y),r:p.event===selected?9:6,class:'goal'}});c.addEventListener('click',()=>select(p.event))}});mk('text',{{x:12,y:H-12}}).textContent=`map frame · ${{fmt(dx)}} × ${{fmt(dy)}} m`;}}
function summary(e){{const r=e.result||{{}},a=e.arguments||{{}};if(e.tool==='search_goal_images')return `“${{a.query||r.query||''}}” · ${{(r.results||[]).length}} hits`;if(e.tool.includes('preview'))return `${{r.capture_ref||a.capture_ref||''}} · ${{fmt(r.planned_path_distance_m)}} m planned`;if(e.tool==='navigate_to_image')return `${{r.capture_ref||''}} · action ${{r.action_id||'—'}}`;if(e.tool==='get_navigation_status')return `${{r.phase||r.navigator_status||'status'}} · ${{fmt(r.distance_traveled_m)}} m`;if(e.tool==='get_navigation_report')return `${{r.phase||'report'}} · ${{fmt(r.distance_traveled_m)}} m · ${{r.selected_frame_count??0}} frames`;if(e.tool==='get_pose_on_map')return `${{r.view||a.view||'map'}} · pose ${{r.pose?`${{fmt(r.pose.x)}}, ${{fmt(r.pose.y)}}`:'—'}}`;return r.reason||r.message||e.status}}
function detail(e){{const r=e.result||{{}},facts=[['elapsed',`+${{e.elapsed_s.toFixed(1)}} s`],['status',e.status],['lane',e.lane]];[['capture',r.capture_ref],['action',r.action_id],['phase',r.phase],['planned',r.planned_path_distance_m!=null?fmt(r.planned_path_distance_m)+' m':null],['traveled',r.distance_traveled_m!=null?fmt(r.distance_traveled_m)+' m':null],['cross-track',r.cross_track_error_m!=null?fmt(r.cross_track_error_m)+' m':null]].forEach(x=>{{if(x[1]!=null)facts.push(x)}});document.getElementById('detail').innerHTML=`<div class="eyebrow">event ${{e.index}} · ${{new Date(e.timestamp).toLocaleTimeString()}}</div><h3 style="color:${{colors[e.lane]}}">${{esc(e.tool)}}</h3><div class="sub">${{esc(summary(e))}}</div>${{e.image?`<img src="${{e.image}}" alt="Visual output from ${{esc(e.tool)}}">`:''}}<div class="facts">${{facts.map(x=>`<div class="fact"><span>${{esc(x[0])}}</span>${{esc(x[1])}}</div>`).join('')}}</div><details><summary>Arguments</summary><pre>${{esc(JSON.stringify(e.arguments,null,2))}}</pre></details><details><summary>Result</summary><pre>${{esc(JSON.stringify(e.result,null,2))}}</pre></details>`}}
function drawTimeline(){{const root=document.getElementById('timeline'),lanes=['memory','planning','movement','evidence','state'],dur=Math.max(1,run.duration_s);root.innerHTML='<div class="axis"></div>';const axis=root.querySelector('.axis');for(let i=0;i<=5;i++){{const t=dur*i/5;axis.insertAdjacentHTML('beforeend',`<span class="tick" style="left:${{i*20}}%">+${{t.toFixed(0)}}s</span>`)}}lanes.forEach(l=>{{const row=document.createElement('div');row.className='lane';row.dataset.lane=l;row.innerHTML=`<span class="lane-label">${{l}}</span>`;run.events.forEach((e,i)=>{{if(e.lane!==l)return;const b=document.createElement('button');b.className='event '+(i===selected?'sel ':'')+(e.status==='failed'?'failed':'');b.style.cssText=`left:${{e.elapsed_s/dur*100}}%;--c:${{colors[l]}}`;b.title=`${{e.tool}} · +${{e.elapsed_s.toFixed(1)}}s`;b.textContent=e.index;b.onclick=()=>select(i);row.appendChild(b)}});root.appendChild(row)}})}}
function renderList(){{document.getElementById('event-list').innerHTML=run.events.map((e,i)=>active.has(e.lane)?`<button class="event-card ${{i===selected?'sel':''}}" style="--c:${{colors[e.lane]}}" data-i="${{i}}"><span class="num">#${{e.index}} · +${{e.elapsed_s.toFixed(1)}} s · ${{e.lane}}</span><strong>${{esc(e.tool)}}</strong><p>${{esc(summary(e))}}</p></button>`:'').join('');document.querySelectorAll('.event-card').forEach(b=>b.onclick=()=>select(+b.dataset.i))}}
function filters(){{document.getElementById('filters').innerHTML=Object.keys(colors).map(l=>`<button class="filter" data-l="${{l}}" aria-pressed="${{active.has(l)}}"><i class="sw" style="--c:${{colors[l]}}"></i>${{l}}</button>`).join('');document.querySelectorAll('.filter').forEach(b=>b.onclick=()=>{{active.has(b.dataset.l)?active.delete(b.dataset.l):active.add(b.dataset.l);filters();renderList()}})}}
function select(i){{selected=i;detail(run.events[i]);drawMap();drawTimeline();renderList()}} filters();select(0);new ResizeObserver(drawMap).observe(document.getElementById('map'));
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="rollout JSONL, thread UUID, or codex:// URL")
    parser.add_argument("-o", "--output", type=pathlib.Path,
                        help="output HTML (default: spatial-memory-<thread>.html)")
    parser.add_argument("--sessions-dir", type=pathlib.Path,
                        default=pathlib.Path.home() / ".codex" / "sessions")
    parser.add_argument("--no-images", action="store_true",
                        help="omit embedded visual evidence for a smaller file")
    args = parser.parse_args(argv)
    try:
        rollout = resolve_rollout(args.source, args.sessions_dir.expanduser())
        data = load_timeline(rollout, include_images=not args.no_images)
        output = args.output or pathlib.Path(
            f"spatial-memory-{data['thread_id']}.html"
        )
        output = output.expanduser().resolve()
        output.write_text(render_html(data), encoding="utf-8")
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
