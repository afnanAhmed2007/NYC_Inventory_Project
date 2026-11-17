"""Build the two-page CS 210 proposal PDF."""

from pathlib import Path

from fpdf import FPDF

FONT = Path("/usr/share/fonts/truetype/liberation")
OUT = Path(__file__).resolve().parents[1] / "NYC-Infrastructure-Intelligence-Proposal.pdf"

# Word-like single spacing for 12 pt type: about 15 pt between baselines.
BODY = 12
LEAD = 14
TITLE = 14
TITLE_LEAD = 18


class Proposal(FPDF):
    def footer(self):
        self.set_y(-36)
        self.set_font("Serif", "", 10)
        self.set_text_color(70, 70, 70)
        self.cell(0, 12, str(self.page_no()), align="C")


def paragraph(pdf: Proposal, text: str, gap: float = 4) -> None:
    pdf.set_font("Serif", "", BODY)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, LEAD, text, align="J", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(gap)


def heading(pdf: Proposal, text: str) -> None:
    pdf.ln(2)
    pdf.set_font("Serif", "B", BODY)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, LEAD, text, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def build() -> Proposal:
    pdf = Proposal(format="Letter", unit="pt")
    pdf.set_auto_page_break(auto=True, margin=46)
    pdf.set_margins(72, 64, 72)
    pdf.set_title(
        "NYC Infrastructure Intelligence: Predicting Street-Maintenance Demand and Unresolved Requests"
    )
    pdf.set_author("Afnan Ahmed")
    pdf.add_font("Serif", "", str(FONT / "LiberationSerif-Regular.ttf"))
    pdf.add_font("Serif", "B", str(FONT / "LiberationSerif-Bold.ttf"))
    pdf.add_font("Serif", "I", str(FONT / "LiberationSerif-Italic.ttf"))
    pdf.add_page()

    pdf.set_font("Serif", "B", TITLE)
    pdf.multi_cell(
        0,
        TITLE_LEAD,
        "NYC Infrastructure Intelligence: Predicting Street-Maintenance Demand and Unresolved Requests",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(4)
    pdf.set_font("Serif", "", BODY)
    pdf.multi_cell(0, LEAD, "Afnan Ahmed", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(
        0,
        LEAD,
        "CS 210: Data Management in Data Science",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.multi_cell(
        0, LEAD, "November 16, 2026", align="C", new_x="LMARGIN", new_y="NEXT"
    )
    pdf.ln(6)

    heading(pdf, "1. Project Definition")
    paragraph(
        pdf,
        "New York City often learns a street has failed only after a resident files a 311 request. The Department of Transportation (DOT) and borough maintenance crews then decide where to send a limited number of workers. I will build a system that, for a chosen week and any of the five boroughs, shows which community districts are likely to generate the most street-maintenance requests, which measured conditions travel with that load, and where people have already asked for repairs that remain unfinished.",
    )
    paragraph(
        pdf,
        "The scale is documented in DOT's Fiscal Year 2026 Mayor's Management Report. DOT maintains about 6,300 miles of streets, highways, and plazas, 809 bridges, more than 13,500 signalized intersections, and 315,000 street lights. In FY2026, 73.8 percent of surveyed streets were rated Good, above a 71 percent target, while 25.8 percent were Fair and 0.5 percent were Poor. Crews repaired 229,365 potholes, up 27 percent from the year before, an increase DOT linked in part to freeze-thaw weather. The average time to close a 311 pothole request rose from 1.99 days to 4.0 days. That average stayed under a five-day target and excludes four proactive repair blitzes. Resurfacing held near 1,169 lane miles at about $244,000 per lane mile. A citywide figure hides the district about to absorb the next hard weather week.",
    )
    paragraph(
        pdf,
        "The project follows the three parts of CS 210. SQL stores, updates, and queries complaints, districts, agencies, pavement, population, weather, and model scores. Data science collects and cleans the public files, builds a district-week table, and draws the maps and charts. Machine learning forecasts weekly load and estimates which requests will stay unresolved.",
        gap=2,
    )

    heading(pdf, "2. Novelty and Importance")
    paragraph(
        pdf,
        "I chose this project because the decision belongs to a working agency and the files are public. A crew needs a job count. An analyst also needs a rate per 10,000 residents, so a large district is not ranked first only because more people live there, and an unresolved share, so a quieter district that misses deadlines stays visible.",
    )
    paragraph(
        pdf,
        "The city already publishes the pieces separately. The 311 portal lists requests after they arrive. DOT's Street Pavement Ratings map stores the latest survey score of each segment. The Mayor's Management Report publishes the citywide totals above, after the fiscal year. Those products describe the past. They leave weather, pavement, population, and complaint history in separate places, and they do not store a forecast for a chosen week next to requests that are already past due.",
    )
    paragraph(
        pdf,
        "A report is not a full inventory: districts that use 311 more will look busier, a pavement rating is only the last inspection, and one Central Park series cannot see a storm that hits one borough harder than another. DOT's Priority Investment Areas already guide longer-term capital work, and this project leaves that program as it is. The question here is where crews should go in a given week, and where requests are already late.",
        gap=2,
    )

    heading(pdf, "3. Data and Methodology")
    paragraph(
        pdf,
        "Five public sources feed one warehouse. Requests come from NYC Open Data 311 (erm2-nwe9), from January 1, 2022 through the latest complete week: street, sidewalk, highway, curb, and bridge conditions, street lights, traffic signals, and damaged or missing signs. Potholes stay a descriptor of street condition. Central Park weather from Open-Meteo becomes weekly freeze-thaw days, rain, and snow, labeled as citywide. Street Pavement Ratings (6yyb-pb25) give a 1-to-10 score, length, and inspection date. American Community Survey five-year population (table B01003) is summed from tracts to community districts with the City Planning equivalency, and the survey year is stored. District boundaries supply the map and the area used for density.",
    )
    paragraph(
        pdf,
        "SQLite is the warehouse, with foreign keys and indexes on district, type, and date. Tables cover borough, community district, agency, complaint type, complaint, weather, pavement, population, weekly demand, load forecast, and unfixed score. Extracts land in staging, and a later run rebuilds derived tables so a correction replaces a stale row. For a chosen week, queries return counts and rates per 10,000 residents by district and borough, pothole volume against freeze-thaw days, fair-or-poor mileage, past-due requests, same-street complaints that return within 90 days, and week-over-week change. Stored forecasts let one query list districts that are both busy and already behind.",
    )
    paragraph(
        pdf,
        "Cleaning is part of the analysis. \"12 MANHATTAN\" maps to the official district code. Unspecified boards, missing dates, and negative durations are dropped and counted. A request is unfixed if it is still open past its due date, or past 14 days when no due date exists. Empty weeks are stored as zeros. Segments use the file's own groups: good 8 to 10, fair from 4 up to 8, and poor below 4. Each model row is one district-week, with lags, last year's same week, weather, pavement share, population, density, and a backlog from older requests only.",
    )
    paragraph(
        pdf,
        "Two tabular models hold out the last 12 complete weeks. The load model predicts the weekly count and the per-capita rate against the same week last year, or the trailing four-week mean. Poisson regression fits the count from week-of-year, borough, freeze-thaw days, precipitation, snow, fair-or-poor share, population, and last year's count; its coefficients are the factor readout. Histogram gradient boosting adds lags and district identity and is scored with mean absolute error. A borough forecast sums its districts. The second model predicts a left-unfixed request against that type's historical rate. Logistic regression uses type, a pothole flag, borough, pavement, opening-week weather, month, and morning backlog; gradient boosting is the comparison. I report ROC-AUC and average precision, because accuracy would flatter a model that misses this minority class.",
        gap=2,
    )

    heading(pdf, "4. Implementation and Evaluation Plan")
    paragraph(
        pdf,
        "Work follows the data, then the models, then the screen: cache each source with its vintage, clean identifiers, audit dropped rows, load the tables, run the queries, train both models, insert the scores, and draw the map. A fixture tests the loader first. Checks cover the district crosswalk, population sums, pavement weights, negative durations, the unfixed and came-back rules, zero-filled weeks, foreign keys, a known borough total, and the screen itself: each borough, a past week with actual counts beside the prediction, and an upcoming week labeled with last year's weather.",
    )
    paragraph(
        pdf,
        "The project is successful when three checks hold. For a chosen week and borough, SQL returns the count, the rate per 10,000 residents, weather, pavement, and the unfixed and came-back totals. The holdout reports mean absolute error against the seasonal baseline and average precision against the type-rate baseline, even if the model does not win. For one borough and week, the screen shows where work is likely, which factors were used, and which districts are past due. The limits stay in the write-up: 311 counts reports, Central Park stands in for the city, and a pavement score dates from its inspection.",
        gap=0,
    )
    return pdf


if __name__ == "__main__":
    pdf = build()
    pdf.output(str(OUT))
    print(f"pages={pdf.pages_count} -> {OUT}")
