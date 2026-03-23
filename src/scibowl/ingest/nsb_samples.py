from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen


HS_SAMPLE_QUESTIONS_URL = "https://science.osti.gov/wdts/nsb/Regional-Competitions/Resources/HS-Sample-Questions"


@dataclass
class SamplePacketLink:
    set_number: int
    label: str
    url: str


class HSSampleQuestionsParser(HTMLParser):
    set_pattern = re.compile(r"Sample Questions? Set\s+(\d+)|Sample Question Set\s+(\d+)", re.IGNORECASE)

    def __init__(self, *, min_set: int) -> None:
        super().__init__()
        self.min_set = min_set
        self.current_set: int | None = None
        self.current_href: str | None = None
        self.current_anchor_text: list[str] = []
        self.links: list[SamplePacketLink] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            attr_map = dict(attrs)
            self.current_href = attr_map.get("href")
            self.current_anchor_text = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return

        match = self.set_pattern.search(text)
        if match:
            set_number = next(int(group) for group in match.groups() if group)
            self.current_set = set_number

        if self.current_href is not None:
            self.current_anchor_text.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a":
            return

        href = self.current_href
        label = " ".join(self.current_anchor_text).strip()
        if (
            href
            and self.current_set is not None
            and self.current_set >= self.min_set
            and label.lower().startswith("round")
        ):
            self.links.append(
                SamplePacketLink(
                    set_number=self.current_set,
                    label=label,
                    url=urljoin(HS_SAMPLE_QUESTIONS_URL, href),
                )
            )

        self.current_href = None
        self.current_anchor_text = []


def scrape_hs_sample_packet_links(*, min_set: int = 13) -> list[SamplePacketLink]:
    with urlopen(HS_SAMPLE_QUESTIONS_URL) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = HSSampleQuestionsParser(min_set=min_set)
    parser.feed(html)
    return parser.links


def download_sample_packets(output_dir: Path, *, min_set: int = 13) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    links = scrape_hs_sample_packet_links(min_set=min_set)
    written: list[Path] = []

    for link in links:
        destination = output_dir / canonical_packet_filename(link)
        with urlopen(link.url) as response:
            destination.write_bytes(response.read())
        written.append(destination)

    return written


def canonical_packet_filename(link: SamplePacketLink) -> str:
    round_match = re.search(r"round\s+(\d+)", link.label, re.IGNORECASE)
    if round_match:
        return f"set_{link.set_number:02d}__round_{int(round_match.group(1)):02d}.pdf"

    parsed = urlparse(link.url)
    source_name = Path(parsed.path).name
    return f"set_{link.set_number:02d}__{source_name}"
