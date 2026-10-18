"""Read-only connectivity/schema and Task 2 profile check; no model key needed."""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hw2 import _json_value, load_mcp_tools


async def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    tools = await asyncio.wait_for(load_mcp_tools(), timeout=60)
    print("MCP tools:", ", ".join(tool.name for tool in tools))
    lookup = {tool.name: tool for tool in tools}
    data = {"schemas": {tool.name: tool.args for tool in tools}}
    for name, args in [
        ("get_linkedin_profile", {"person_id": 10001}),
        ("get_facebook_profile", {"user_id": 10001}),
    ]:
        content = await asyncio.wait_for(lookup[name].ainvoke(args), timeout=45)
        data[name] = _json_value(content)
    output = ROOT / "artifacts" / "mcp_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
