


def _resolve_package_versions(requirement: str) -> dict[str, str]:
    from pip_requirements_parser import RequirementsFile

    requirements_txt = RequirementsFile.from_file(
        requirement,
        include_nested=True,
    )
    deps: dict[str, str] = {}
    for req in requirements_txt.requirements:
        if (
            req.is_editable
            or req.is_local_path
            or req.is_url
            or req.is_wheel
            or not req.name
            or not req.specifier
        ):
            continue
        for sp in req.specifier:
            if sp.operator == "==":
                assert req.line is not None
                deps[req.name] = req.line
                break
    return deps
