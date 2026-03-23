from scibowl.ingest.nsb_samples import HSSampleQuestionsParser, SamplePacketLink, canonical_packet_filename


def test_hs_sample_questions_parser_filters_min_set() -> None:
    html = """
    <html><body>
    <p>Sample Questions Set 12</p>
    <a href="/set12/r1.pdf">Round 1</a>
    <p>Sample Questions Set 13</p>
    <a href="/set13/r1.pdf">Round 1</a>
    <a href="/set13/r2.pdf">Round 2</a>
    <p>Sample Questions Set 14</p>
    <a href="/set14/r7.pdf">Round 7</a>
    <a href="/other/page">Sample Question Set 15</a>
    </body></html>
    """

    parser = HSSampleQuestionsParser(min_set=13)
    parser.feed(html)

    assert len(parser.links) == 3
    assert [link.set_number for link in parser.links] == [13, 13, 14]


def test_canonical_packet_filename_uses_set_and_round() -> None:
    filename = canonical_packet_filename(
        SamplePacketLink(set_number=13, label="Round 6", url="https://example.com/weird-name.pdf")
    )

    assert filename == "set_13__round_06.pdf"
