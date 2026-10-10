#!/usr/bin/env python3
"""Generate Meridian's device dispatch table from the Khronos registry.

    python code/gen_dispatch.py --day 02        write into snapshots/day02/src
    python code/gen_dispatch.py --list          what is wanted, and what level each call is

WHAT vk.xml IS. The Vulkan registry: every command, every structure, every enum and every
extension in Vulkan, machine-readable, with the version each arrived in. It is what the C
headers are generated from, what every language binding is generated from, and what the
specification's own tooling reads. It is not a third-party file -- it IS the API, in a form a
program can read. This book keeps a copy at __shared/_docs/libraries/vulkan/registry/vk.xml.

WHY GENERATE A DISPATCH TABLE AT ALL, when vulkan-1.lib already links.

Every Vulkan command has a LEVEL. `vkCreateInstance` is global; `vkEnumeratePhysicalDevices`
takes a VkInstance and is instance-level; `vkCmdDraw` takes a VkCommandBuffer and is
device-level. When you call vkCmdDraw through vulkan-1.lib, you are calling the LOADER's
vkCmdDraw -- a trampoline, which reads a dispatch pointer out of the handle you passed and
jumps to the driver's real function.

For one call that is nothing. For a frame that records ten thousand of them it is ten thousand
extra loads and jumps, and the loader's own documentation recommends avoiding it. Fetching each
device-level function once with vkGetDeviceProcAddr and calling it directly removes the hop.

WHAT IT GENERATES. One struct of function pointers, filled in from vkGetDeviceProcAddr, and a
macro per command so the calls in the engine read exactly as they did before. WANTED below is
the list Meridian actually calls -- it grows by a line or two most days -- and the generated
file stays short enough to open and read, which is the point of having written it yourself.
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK = HERE.parent

DEFAULT_XML = next(
    (p / "__shared" / "_docs" / "libraries" / "vulkan" / "registry" / "vk.xml"
     for p in BOOK.parents
     if (p / "__shared" / "_docs" / "libraries" / "vulkan" / "registry" / "vk.xml").is_file()),
    None)

# ---- what Meridian calls through the device, grouped the way the engine thinks ------------
# INSTANCE-LEVEL AND GLOBAL CALLS ARE NOT HERE. vkCreateInstance, vkEnumeratePhysicalDevices
# and vkCreateDevice happen once each; a trampoline on a call made once is not worth a line of
# generated code. This table is the calls made per frame, per draw, or per object.
WANTED = {
    "recording": [
        "vkBeginCommandBuffer", "vkEndCommandBuffer", "vkResetCommandBuffer",
        "vkCmdBeginRendering", "vkCmdEndRendering", "vkCmdBindPipeline", "vkCmdDraw",
        "vkCmdPipelineBarrier2", "vkCmdCopyImageToBuffer",
        "vkCmdBindVertexBuffers", "vkCmdBindIndexBuffer", "vkCmdDrawIndexed",
        "vkCmdCopyBuffer", "vkCmdClearColorImage",
        "vkCmdSetViewport", "vkCmdSetScissor", "vkCmdSetLineWidth",
    ],
    "presenting": [
        "vkCreateSwapchainKHR", "vkDestroySwapchainKHR", "vkGetSwapchainImagesKHR",
        "vkAcquireNextImageKHR", "vkQueuePresentKHR", "vkCreateSemaphore",
        "vkDestroySemaphore", "vkCmdBlitImage",
    ],
    # DAY 18 ADDS A WHOLE PIPELINE TYPE, and it is two entry points. A compute pipeline
    # is created by its own call and run by its own command, and every other thing it
    # needs -- layouts, descriptors, barriers, push constants -- is already in this table.
    "compute": [
        "vkCreateComputePipelines", "vkCmdDispatch",
    ],
    "submission": [
        "vkQueueSubmit2", "vkQueueWaitIdle", "vkWaitForFences", "vkResetFences",
        "vkDeviceWaitIdle",
    ],
    "objects": [
        "vkCreateImage", "vkDestroyImage", "vkCreateImageView", "vkDestroyImageView",
        "vkCreateBuffer", "vkDestroyBuffer", "vkAllocateMemory", "vkFreeMemory",
        "vkBindImageMemory", "vkBindBufferMemory", "vkMapMemory", "vkUnmapMemory",
        "vkGetImageMemoryRequirements", "vkGetBufferMemoryRequirements",
        "vkCreateGraphicsPipelines", "vkDestroyPipeline", "vkCreatePipelineLayout",
        "vkDestroyPipelineLayout", "vkCreateShaderModule", "vkDestroyShaderModule",
        "vkCreateCommandPool", "vkDestroyCommandPool", "vkAllocateCommandBuffers",
        "vkFreeCommandBuffers", "vkCreateFence", "vkDestroyFence",
        "vkCreatePipelineCache", "vkDestroyPipelineCache", "vkGetPipelineCacheData",
    ],
}

# WHEN EACH CALL ARRIVES IN THE ENGINE. A snapshot is the engine as it stood at the end of one
# day, so the table inside dayNN holds what the engine could call on day NN and nothing later.
# Anything absent has been in the table since Day 2, when it replaced the loader's trampoline.
SINCE: dict[str, int] = {
    "vkCreateSwapchainKHR": 3, "vkDestroySwapchainKHR": 3, "vkGetSwapchainImagesKHR": 3,
    "vkAcquireNextImageKHR": 3, "vkQueuePresentKHR": 3, "vkCreateSemaphore": 3,
    "vkDestroySemaphore": 3, "vkCmdBlitImage": 3,
    # DAY 4 ADDED FIVE CALLS AND NOBODY UPDATED THE TABLE. They worked -- an entry point that
    # is not in the table simply goes through the loader's trampoline, which is what every
    # Vulkan call did before Day 2. The table is a list you maintain, and nothing tells you
    # when it has fallen behind the engine.
    "vkCmdBindVertexBuffers": 5, "vkCmdBindIndexBuffer": 5, "vkCmdDrawIndexed": 5,
    "vkCmdCopyBuffer": 5, "vkCmdClearColorImage": 5,
    # and today's own five
    "vkCmdSetViewport": 5, "vkCmdSetScissor": 5, "vkCmdSetLineWidth": 5,
    "vkCreatePipelineCache": 5, "vkDestroyPipelineCache": 5, "vkGetPipelineCacheData": 5,
    "vkCreateComputePipelines": 18, "vkCmdDispatch": 18,
}
FIRST_GENERATED_DAY = 2


def wanted_on(day: int) -> dict:
    out = {}
    for group, members in WANTED.items():
        keep = [n for n in members if SINCE.get(n, FIRST_GENERATED_DAY) <= day]
        if keep:
            out[group] = keep
    return out


def load_registry(xml_path: Path):
    """(return type, parameter list, first parameter's type) for every command in vk.xml."""
    root = ET.parse(xml_path).getroot()
    sigs: dict[str, tuple[str, str, str]] = {}
    aliases: dict[str, str] = {}
    for cmd in root.iter("command"):
        capi = cmd.get("api")
        if capi and "vulkan" not in capi.split(","):
            continue
        if cmd.get("alias"):
            aliases[cmd.get("name")] = cmd.get("alias")
            continue
        proto = cmd.find("proto")
        if proto is None:
            continue
        name_el = proto.find("name")
        if name_el is None:
            continue
        name = name_el.text
        ret = "".join(proto.itertext())[: -len(name)].strip() or "void"
        params, first = [], ""
        for p in cmd.findall("param"):
            # vk.xml DESCRIBES TWO APIs. Vulkan and Vulkan SC -- the Safety Critical profile --
            # share one registry, and a few parameters are listed TWICE with an api attribute
            # saying which is which. vkCreateSwapchainKHR is one: taking both gives a function
            # with two parameters called pCreateInfo, which is a redefinition error in a header
            # generated from the API's own description of itself.
            api = p.get("api")
            if api and "vulkan" not in api.split(","):
                continue
            pname_el = p.find("name")
            if pname_el is None:
                continue
            pname = pname_el.text
            ptype = "".join(p.itertext())[: -len(pname)].strip()
            params.append(f"{ptype} {pname}")
            if not first:
                t = p.find("type")
                first = t.text if t is not None else ""
        sigs[name] = (ret, ", ".join(params) if params else "void", first)
    # An alias is the same function under a second name -- vkQueueSubmit2KHR is
    # vkQueueSubmit2. Resolving them means the table can be written with core names.
    for a, target in aliases.items():
        if target in sigs:
            sigs[a] = sigs[target]
    return sigs


# THE FIRST PARAMETER DECIDES THE LEVEL, and the registry states it plainly: a command whose
# first parameter is a VkDevice, VkQueue or VkCommandBuffer is DEVICE-level and can be fetched
# from vkGetDeviceProcAddr. Anything else cannot, and asking for it returns null.
DEVICE_FIRST = {"VkDevice", "VkQueue", "VkCommandBuffer"}


def generate(sigs, wanted) -> tuple[str, str]:
    names = [n for group in wanted.values() for n in group]
    missing = [n for n in names if n not in sigs]
    if missing:
        raise SystemExit("not in the registry: " + ", ".join(missing))
    wrong = [n for n in names if sigs[n][2] not in DEVICE_FIRST]
    if wrong:
        raise SystemExit("not device-level, so vkGetDeviceProcAddr cannot return them: "
                         + ", ".join(wrong))

    h = ["// GENERATED by code/gen_dispatch.py from the Khronos registry. Do not edit by hand.",
         "//",
         "// Every function below is fetched once with vkGetDeviceProcAddr and called through",
         "// this table afterwards. Calling it through vulkan-1.lib instead reaches the LOADER's",
         "// copy, which reads a dispatch pointer out of the first argument and jumps -- correct,",
         "// and one indirection per call that a program recording thousands of draws can skip.",
         "//",
         "// The registry decides what is in here: a command is device-level exactly when its",
         "// first parameter is a VkDevice, a VkQueue or a VkCommandBuffer.",
         "#pragma once",
         "#include <vulkan/vulkan.h>",
         "",
         "struct DeviceDispatch {"]
    c = ['// GENERATED by code/gen_dispatch.py from the Khronos registry. Do not edit by hand.',
         "#define VK_DISPATCH_NO_MACROS",
         '#include "vk_dispatch.h"',
         "",
         "#include <cstdio>",
         "",
         "int load_device_dispatch(VkDevice device, DeviceDispatch& d) {",
         "    int missing = 0;",
         "    auto get = [&](const char* name) {",
         "        PFN_vkVoidFunction f = vkGetDeviceProcAddr(device, name);",
         "        if (!f) { std::printf(\"  MISSING %s\\n\", name); ++missing; }",
         "        return f;",
         "    };"]

    count = 0
    for group, members in wanted.items():
        h.append(f"    // ---- {group} " + "-" * max(0, 62 - len(group)))
        c.append(f"    // ---- {group} " + "-" * max(0, 62 - len(group)))
        for name in members:
            ret, params, _first = sigs[name]
            h.append(f"    {ret} (VKAPI_PTR* {name})({params});")
            c.append(f"    d.{name} = (PFN_{name})get(\"{name}\");")
            count += 1
    h += ["};",
          "",
          "// Fills the table, prints anything the driver does not have, and returns how many.",
          "int load_device_dispatch(VkDevice device, DeviceDispatch& d);",
          "",
          "// THE TABLE IS A GLOBAL, and that is a deliberate simplification: this engine has",
          "// one device for its whole life. A program with two would carry one table each.",
          "extern DeviceDispatch vk;",
          f"""static const int DEVICE_DISPATCH_COUNT = {count};""",
          "",
          "// SO THAT THE ENGINE'S CALL SITES READ AS THEY ALWAYS DID. A macro per command,",
          "// which is the whole of what makes the switch from the loader's trampoline to this",
          "// table invisible in main.cpp.",
          "//",
          "// EACH MACRO IS FUNCTION-LIKE, AND THAT IS NOT A STYLE CHOICE. An object-like",
          "// macro expands wherever the name appears -- including in (::vkCmdDraw), which is",
          "// how Day 2's benchmark reaches the LOADER's copy for comparison. A function-like",
          "// macro expands only when the name is followed by an open bracket.",
          "//",
          "// THE GUARD IS NOT DECORATION EITHER. vk_dispatch.cpp assigns to d.vkCmdDraw, and",
          "// the generated source defines VK_DISPATCH_NO_MACROS before including this file.",
          "#ifndef VK_DISPATCH_NO_MACROS",
          ]
    for group, members in wanted.items():
        for name in members:
            h.append(f"#define {name}(...) vk.{name}(__VA_ARGS__)")
    h.append("#endif")
    h.append("")

    c += ["    return missing;",
          "}",
          "",
          "DeviceDispatch vk = {};"]
    return "\n".join(h), "\n".join(c), count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--day", type=int)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if not a.xml or not a.xml.is_file():
        raise SystemExit("vk.xml not found; pass --xml")
    sigs = load_registry(a.xml)
    print(f"vk.xml: {len(sigs)} commands")

    wanted = wanted_on(a.day) if a.day else WANTED
    if a.list:
        for group, members in wanted.items():
            print(f"  {group}")
            for n in members:
                print(f"    {n:<38} first parameter {sigs[n][2]}")
        return 0

    header, source, count = generate(sigs, wanted)
    if a.day is None:
        raise SystemExit("say --day NN")
    out = BOOK / "code" / "snapshots" / f"day{a.day:02d}" / "src"
    out.mkdir(parents=True, exist_ok=True)
    (out / "vk_dispatch.h").write_text(header, encoding="utf-8", newline="\n")
    (out / "vk_dispatch.cpp").write_text(source, encoding="utf-8", newline="\n")
    print(f"wrote {out}/vk_dispatch.h and .cpp -- {count} device-level entry points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
