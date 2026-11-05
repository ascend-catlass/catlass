"""Small MkDocs hooks used by the CATLASS documentation site."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import yaml

from mkdocs.plugins import event_priority
from mkdocs.structure.files import File, InclusionLevel


_MARKDOWN_LINK = re.compile(
    r"(?P<prefix>\]\(\s*<?)"
    r"(?P<target>(?:\.\./)+[^)\s>]+)"
    r"(?P<suffix>>?(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\))"
)

_PAGE_NAV_TITLES = {
    "zh": {"README.md": "主页", "FAQ.md": "FAQ"},
    "en": {"README.md": "Home", "FAQ.md": "FAQ"},
}

_TOP_LEVEL_ORDER = {
    "README.md": 0,
    "1_Practice": 10,
    "2_Design": 20,
    "3_API": 30,
    "4_CATLASS_Python": 40,
    "0x_new_versions": 50,
    "FAQ.md": 100,
}

_ASSETS_PATH = Path("assets")
_DSL_DOCS_PATH = Path("python/tla_dsl/docs")
_DSL_SITE_SECTION = Path("4_CATLASS_Python")
_DSL_GENERATED_BY = "mkdocs_hooks.dsl_docs"

_DSL_DEFAULT_ORDER = 90


def _asset_sources(assets_root: Path):
    """Yield centralized assets and their public paths."""
    for source in assets_root.rglob("*"):
        if source.is_file():
            yield _ASSETS_PATH / source.relative_to(assets_root), source


def _dsl_sources(dsl_root: Path, locale: str):
    """Yield canonical DSL Markdown sources and their virtual site paths."""
    locale_root = dsl_root / locale
    if not locale_root.is_dir():
        return

    for source in locale_root.rglob("*.md"):
        virtual_source = _DSL_SITE_SECTION / source.relative_to(locale_root)
        yield virtual_source, source


def _file_destination(path: Path, config, output_prefix: Path) -> Path:
    """Return MkDocs' normal HTML destination with a locale prefix."""
    probe = File(
        path.as_posix(),
        src_dir=None,
        dest_dir=config.site_dir,
        use_directory_urls=config.use_directory_urls,
    )
    return output_prefix / probe.dest_uri


@event_priority(-200)
def on_files(files, config, **kwargs):
    """Expose centralized assets and canonical DSL docs in each locale build."""
    assets_root = Path(config.docs_dir) / _ASSETS_PATH
    i18n = config.plugins["i18n"]
    locale = i18n.current_language
    default_locale = i18n.default_language

    # The i18n plugin filters root-level assets when document fallback is
    # disabled. Replace its partial view with virtual, locale-aware files.
    output_prefix = Path() if locale == default_locale else Path(locale)

    if assets_root.is_dir():
        for file in list(files):
            if not file.abs_src_path:
                continue

            try:
                Path(file.abs_src_path).resolve().relative_to(assets_root.resolve())
            except ValueError:
                continue

            files.remove(file)

        for public_path, source in _asset_sources(assets_root):
            virtual_source = Path(locale) / public_path
            destination = output_prefix / public_path
            asset = File(
                virtual_source.as_posix(),
                src_dir=None,
                dest_dir=config.site_dir,
                use_directory_urls=config.use_directory_urls,
                dest_uri=destination.as_posix(),
            )
            asset.abs_src_path = str(source)
            asset.generated_by = "mkdocs_hooks"
            files.append(asset)

    project_root = Path(config.config_file_path).parent.resolve()
    dsl_root = project_root / _DSL_DOCS_PATH
    for virtual_source, source in _dsl_sources(dsl_root, locale):
        destination = _file_destination(virtual_source, config, output_prefix)
        localized_source = Path(locale) / virtual_source
        dsl_file = File(
            localized_source.as_posix(),
            src_dir=None,
            dest_dir=config.site_dir,
            use_directory_urls=config.use_directory_urls,
            dest_uri=destination.as_posix(),
            inclusion=InclusionLevel.INCLUDED,
        )
        dsl_file.abs_src_path = str(source)
        dsl_file.generated_by = _DSL_GENERATED_BY
        dsl_file.alternates = {locale: dsl_file}
        dsl_file.locale = locale
        dsl_file.locale_alternate_of = locale
        dsl_file.localization = locale
        dsl_file.norm_src_uri = virtual_source.as_posix()
        files.append(dsl_file)

    return files


def _source_url(target: str, page, config) -> str:
    """Map a link that escapes docs_dir to the corresponding repository URL."""
    parts = urlsplit(target)
    if parts.scheme or parts.netloc:
        return target

    project_root = Path(config.config_file_path).parent.resolve()
    docs_root = Path(config.docs_dir).resolve()
    source_file = Path(page.file.abs_src_path)
    destination = (source_file.parent / unquote(parts.path)).resolve()
    dsl_root = project_root / _DSL_DOCS_PATH

    try:
        source_file.resolve().relative_to(dsl_root)
        source_is_dsl_doc = True
    except ValueError:
        source_is_dsl_doc = False

    try:
        dsl_relative = destination.relative_to(dsl_root)
    except ValueError:
        dsl_relative = None

    if dsl_relative is not None:
        if source_is_dsl_doc:
            return target

        relative_parts = dsl_relative.parts
        if relative_parts and relative_parts[0] in _PAGE_NAV_TITLES:
            relative_parts = relative_parts[1:]
        virtual_target = _DSL_SITE_SECTION.joinpath(*relative_parts)
        page_parts = Path(page.file.src_uri).parts
        if page_parts and page_parts[0] in _PAGE_NAV_TITLES:
            page_parts = page_parts[1:]
        page_parent = Path(*page_parts).parent
        mapped_path = Path(os.path.relpath(virtual_target, page_parent)).as_posix()
        return urlunsplit(("", "", mapped_path, parts.query, parts.fragment))

    try:
        docs_relative = destination.relative_to(docs_root)
    except ValueError:
        docs_relative = None

    if docs_relative is not None:
        if not source_is_dsl_doc:
            return target

        relative_parts = docs_relative.parts
        if relative_parts and relative_parts[0] in _PAGE_NAV_TITLES:
            relative_parts = relative_parts[1:]
        virtual_target = Path(*relative_parts)
        page_parts = Path(page.file.src_uri).parts
        if page_parts and page_parts[0] in _PAGE_NAV_TITLES:
            page_parts = page_parts[1:]
        page_parent = Path(*page_parts).parent
        mapped_path = Path(os.path.relpath(virtual_target, page_parent)).as_posix()
        return urlunsplit(("", "", mapped_path, parts.query, parts.fragment))

    try:
        relative_path = destination.relative_to(project_root)
    except ValueError:
        return target

    if not destination.exists():
        return target

    branch = config.extra.get("source_branch", "master")
    view = "tree" if destination.is_dir() else "blob"
    repository_url = config.repo_url.rstrip("/")
    repository_path = quote(relative_path.as_posix(), safe="/")
    mapped_path = f"{repository_url}/{view}/{branch}/{repository_path}"
    return urlunsplit(("", "", mapped_path, parts.query, parts.fragment))


def on_page_markdown(markdown, page, config, **kwargs):
    """Rewrite repository-relative source links before MkDocs validates them."""

    def replace(match: re.Match[str]) -> str:
        target = _source_url(match.group("target"), page, config)
        return f"{match.group('prefix')}{target}{match.group('suffix')}"

    return _MARKDOWN_LINK.sub(replace, markdown)


def _nav_items(items):
    """Yield every item in a MkDocs navigation tree."""
    for item in items:
        yield item
        children = getattr(item, "children", None)
        if children:
            yield from _nav_items(children)


def _nav_source(item) -> str:
    """Return a navigation item's first source path, without its locale prefix."""
    current = item
    while getattr(current, "file", None) is None:
        children = getattr(current, "children", None)
        if not children:
            return ""
        current = children[0]

    parts = Path(current.file.src_uri).parts
    if parts and parts[0] in _PAGE_NAV_TITLES:
        parts = parts[1:]
    return "/".join(parts)


def _top_level_sort_key(item):
    source = _nav_source(item)
    first_part = source.split("/", 1)[0]
    return (_TOP_LEVEL_ORDER.get(first_part, 90), source)


def _dsl_relative_source(item) -> str:
    source = _nav_source(item)
    prefix = f"{_DSL_SITE_SECTION.as_posix()}/"
    return source.removeprefix(prefix) if source.startswith(prefix) else ""


def _frontmatter(path: Path) -> dict:
    """Read a Markdown file's YAML frontmatter, if present."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    try:
        data = yaml.safe_load(text[3:end])
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _markdown_title(path: Path) -> str | None:
    """Return the first ATX H1 heading of a Markdown file (doc-driven title)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _dsl_apply_title(item) -> None:
    """Set a DSL nav item's title from its own H1 (pages) or index.md H1 (sections)."""
    file = getattr(item, "file", None)
    if file is not None and file.abs_src_path:
        title = _markdown_title(Path(file.abs_src_path))
        if title:
            item.title = title
        return
    # Section: inherit the title of its index.md child.
    for child in getattr(item, "children", []) or []:
        child_file = getattr(child, "file", None)
        if child_file and Path(child_file.src_uri).name == "index.md":
            title = _markdown_title(Path(child_file.abs_src_path))
            if title:
                item.title = title
            return


def _dsl_nav_order(item) -> int:
    """Return the nav ordering for a DSL nav item from its frontmatter."""
    file = getattr(item, "file", None)
    if file is None or not file.abs_src_path:
        # Section: fall back to its index.md child's ordering.
        for child in getattr(item, "children", []) or []:
            child_file = getattr(child, "file", None)
            if child_file and Path(child_file.src_uri).name == "index.md":
                return _dsl_nav_order(child)
        return _DSL_DEFAULT_ORDER
    meta = _frontmatter(Path(file.abs_src_path))
    order = meta.get("nav_order")
    return order if isinstance(order, (int, float)) else _DSL_DEFAULT_ORDER


def _polish_dsl_nav(item, locale: str) -> None:
    """Order the DSL section and apply doc-driven titles to pages and sections."""
    children = getattr(item, "children", None)
    if not children:
        return

    if _nav_source(item).startswith(f"{_DSL_SITE_SECTION.as_posix()}/"):
        children.sort(key=lambda child: (_dsl_nav_order(child), _nav_source(child)))
        for child in children:
            _dsl_apply_title(child)

    for child in children:
        _polish_dsl_nav(child, locale)


def on_page_context(context, page, config, nav, **kwargs):
    """Polish top-level tabs while retaining fully automatic page discovery."""
    locale = getattr(page.file, "locale", "zh")
    titles = _PAGE_NAV_TITLES.get(locale, {})

    for item in _nav_items(nav.items):
        file = getattr(item, "file", None)
        if file and Path(file.src_uri).name in titles:
            item.title = titles[Path(file.src_uri).name]

    for item in nav.items:
        if _nav_source(item).startswith(f"{_DSL_SITE_SECTION.as_posix()}/"):
            item.title = "CATLASS Python"
        _polish_dsl_nav(item, locale)

    if page.file.generated_by == _DSL_GENERATED_BY and page.file.abs_src_path:
        project_root = Path(config.config_file_path).parent.resolve()
        source_path = Path(page.file.abs_src_path).resolve().relative_to(project_root)
        branch = config.extra.get("source_branch", "master")
        repository_path = quote(source_path.as_posix(), safe="/")
        page.edit_url = f"{config.repo_url.rstrip('/')}/edit/{branch}/{repository_path}"

    nav.items.sort(key=_top_level_sort_key)
    return context
