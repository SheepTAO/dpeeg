"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("dpeeg").rglob("*.py")):
    parts = tuple(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        continue

    doc_path = path.relative_to("dpeeg").with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

    with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:  # 
        nav_file.writelines(nav.build_literate_nav())

