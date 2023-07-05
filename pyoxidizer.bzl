# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entrypoint for using pyoxidizer to package openllm into standalone binary distribution."""

VERSION = VARS["version"]
APP_NAME = "openllm"
DISPLAY_NAME = "OpenLLM"
AUTHOR = "BentoML"

def make_msi(target_triple):
    if target_triple == "x86_64-pc-windows-msvc":
        arch = "x64"
    elif target_triple == "i686-pc-windows-msvc":
        arch = "x86"
    else:
        arch = "unknown"

    # https://gregoryszorc.com/docs/pyoxidizer/main/tugger_starlark_type_wix_msi_builder.html
    msi = WiXMSIBuilder(
        id_prefix = APP_NAME,
        product_name = DISPLAY_NAME,
        product_version = VERSION,
        product_manufacturer = AUTHOR,
        arch = arch,
    )
    msi.msi_filename = DISPLAY_NAME + "-" + VERSION + "-" + arch + ".msi"
    msi.help_url = "https://github.com/bentoml/OpenLLM/"
    msi.license_path = CWD + "/LICENSE.md"

    # https://gregoryszorc.com/docs/pyoxidizer/main/tugger_starlark_type_file_manifest.html
    m = FileManifest()

    exe_prefix = "targets/" + target_triple + "/"
    m.add_path(
        path = exe_prefix + APP_NAME + ".exe",
        strip_prefix = exe_prefix,
    )

    msi.add_program_files_manifest(m)

    return msi

def make_exe_installer():
    # https://gregoryszorc.com/docs/pyoxidizer/main/tugger_starlark_type_wix_bundle_builder.html
    bundle = WiXBundleBuilder(
        id_prefix = APP_NAME,
        name = DISPLAY_NAME,
        version = VERSION,
        manufacturer = AUTHOR,
    )

    bundle.add_vc_redistributable("x64")
    bundle.add_vc_redistributable("x86")

    bundle.add_wix_msi_builder(
        builder = make_msi("x86_64-pc-windows-msvc"),
        display_internal_ui = True,
        install_condition = "VersionNT64",
    )
    bundle.add_wix_msi_builder(
        builder = make_msi("i686-pc-windows-msvc"),
        display_internal_ui = True,
        install_condition = "Not VersionNT64",
    )

    return bundle

def make_macos_app_bundle():
    # https://gregoryszorc.com/docs/pyoxidizer/main/tugger_starlark_type_macos_application_bundle_builder.html
    bundle = MacOsApplicationBundleBuilder(DISPLAY_NAME)
    bundle.set_info_plist_required_keys(
        display_name = DISPLAY_NAME,
        identifier = "com.github.bentoml." + APP_NAME,
        version = VERSION,
        signature = "oplm",
        executable = APP_NAME,
    )

    # https://gregoryszorc.com/docs/pyoxidizer/main/tugger_starlark_type_apple_universal_binary.html
    universal = AppleUniversalBinary(APP_NAME)

    for target in ["aarch64-apple-darwin", "x86_64-apple-darwin"]:
        universal.add_path("targets/" + target + "/" + APP_NAME)

    m = FileManifest()
    m.add_file(universal.to_file_content())
    bundle.add_macos_manifest(m)

    return bundle

register_target("windows_installers", make_exe_installer, default = True)
register_target("macos_app_bundle", make_macos_app_bundle)

resolve_targets()
