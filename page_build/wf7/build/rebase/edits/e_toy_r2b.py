"""Review round 2, C13: the role table in "What it means for ThrusterChain" links the page's records where they first
appear, as the part's opening promises (all targets are ids of the page outside the toy section, or #toy-tc-resultf)."""
import rep

P = "toy/part_meaning.html"
rep.apply([
    (P, 'On ThrusterChain so far">The capped pull cost 5.3 to\n          8.8 points',
        'On ThrusterChain so far"><a href="#iw-arms">The capped pull cost 5.3 to\n          8.8 points</a>', 1),
    (P, "The lower edge gained in Stage 2 mostly through a classifier term, and in result F with a supplied",
        "The lower edge gained in <a href=\"#ats-stage2\">Stage 2</a> mostly through a classifier term, and in <a\n"
        "          href=\"#toy-tc-resultf\">result F</a> with a supplied", 1),
    (P, '>Step 1 and step 2 of the restructured BCA told the', '><a href="#ats-restructure">Step 1 and step 2 of the restructured BCA</a> told the', 1),
    (P, "BCA beat plain split conformal by 1.78 points\n          against a 5-point bar.",
        "BCA beat plain split conformal by <a href=\"#ats-routes\">1.78 points</a>\n          against a 5-point bar.", 1),
    (P, ">Weighting restored the band's coverage, but raised", "><a href=\"#iw-results\">Weighting restored the band's coverage</a>, but raised", 1),
])
