# Insightful Investigator

```text

Build a premium, modern, highly polished frontend for an application called:

# FWA RISK INVESTIGATOR

This is an FWA (Fraud, Waste & Abuse) investigation platform.

The frontend should feel like a professional enterprise insurance/payment-integrity product — clean, trustworthy, intelligent, elegant, and presentation-ready.

IMPORTANT:

- Light theme only.

- Use professional colors such as navy, blue, teal, white, soft gray, and subtle risk-status colors.

- Use floating cards, elegant shadows, rounded interfaces, stylish buttons, subtle gradients, and smooth animations.

- Add tasteful animations between pages and on major UI elements.

- Avoid excessive animations, neon colors, clutter, or overly flashy effects.

- Every page must be fully navigable.

- Make the experience feel premium and "Lovable"-style.

- Keep the interface responsive and polished.

==================================================

1. LANDING PAGE

==================================================

Create a visually impressive landing page.

At the center:

FWA RISK INVESTIGATOR

Below the title:

AI-Powered Fraud, Waste & Abuse Investigation

Below that:

[ Enter Secure Portal ]

The title should have a slight floating effect.

The hero area should feel sophisticated with subtle animated analytical/payment-integrity visual elements in the background.

When the user clicks:

Enter Secure Portal

→ Navigate to the Login Page.

==================================================

2. LOGIN PAGE

==================================================

Create a professional secure-login interface.

Title:

SECURE PORTAL

Fields:

Username

User ID

Password

Button:

[ Authenticate ]

Use a clean authentication card with a subtle floating effect.

For now, basic/mock authentication is sufficient.

After successful authentication:

→ Navigate to the Main Page.

==================================================

3. MAIN PAGE

==================================================

Top-center title:

UNCOVER FWA

Subtitle:

AI-Powered Claim & Provider Investigation

Create two large premium floating action cards/buttons:

[ Analyse Claim ]

[ Analyse Provider ]

Each card should contain:

- Appropriate icon

- Short description

- Hover animation

- Elegant shadow

- Professional visual hierarchy

Clicking:

Analyse Claim

→ Uncover Claims

Analyse Provider

→ Uncover Providers

==================================================

4. UNCOVER CLAIMS PAGE

==================================================

Title:

UNCOVER CLAIMS

At the center create two major action cards:

[ Single Claims ]

[ Batch Claims ]

Single Claims

→ Single Claim Analysis Page

Batch Claims

→ Batch Claim Analysis Page

==================================================

5. SINGLE CLAIM ANALYSIS

==================================================

Title:

SINGLE CLAIM ANALYSIS

Create a professional form containing approximately 10 mock claim fields.

Use realistic claim-related fields such as:

Claim ID

Provider ID

Beneficiary ID

Claim Type

Claim Duration

Claim Submitted Amount

Claim Allowed Amount

Claim Payment Amount

Deductible Amount

Service Count

Also include suitable checkbox/toggle fields for claim/status attributes.

Button:

[ Analyse Claim ]

After clicking Analyse Claim:

Show a complete single-claim investigation view containing:

1. Claim Dashboard

2. AI/LLM Explainability Report

3. Human Review section

==================================================

6. SINGLE CLAIM DASHBOARD

==================================================

For the selected claim, show a polished analytical dashboard.

Include cards and visualizations such as:

Risk Score

Risk Rating

Payment vs Peer

Payment Percentile

Service Count

Claim Duration

Claim Frequency

Behavioral Signals

Peer Signals

Use attractive charts and metric cards.

The dashboard should be positioned alongside or above the LLM explanation.

==================================================

7. SINGLE CLAIM LLM EXPLANATION

==================================================

Create a professional report-style panel:

AI INVESTIGATION EXPLANATION

Display:

Risk Score

Risk Rating

Key Signals

Supporting Evidence

Assessment

Reasoning

Example style:

"Potentially Suspicious"

"The claim demonstrates unusual payment and utilization patterns compared with relevant peer behavior."

Do not make the interface imply that the LLM independently proves fraud.

Clearly display:

AI-assisted assessment — Final decision requires domain-expert review.

Below the report:

[ Accept ]

[ Reject ]

==================================================

8. BATCH CLAIM ANALYSIS

==================================================

Title:

BATCH CLAIM ANALYSIS

Provide a large professional upload area:

[ Upload Claims File ]

The user should be able to select a file using the system file explorer.

After a file is uploaded:

Display:

- File name

- Number of records

- Upload status

Then reveal:

[ Analyse ]

When Analyse is clicked:

Display the batch dashboard and results table.

==================================================

9. BATCH CLAIM DASHBOARD

==================================================

At the top display important KPI cards.

Include:

Total Claims

Suspicious Cases

High Risk Cases

Medium Risk Cases

Low Risk Cases

Unique Providers

Unique Beneficiaries

Average Risk Score

Also include useful visualizations:

Risk Distribution

Claim Distribution

Provider Distribution

Payment Distribution

The dashboard represents the complete uploaded dataset.

==================================================

10. BATCH CLAIM RESULTS TABLE

==================================================

Display a table using attributes from the uploaded claim data.

Show relevant columns such as:

Claim ID

Provider ID

Claim Type

Claim Amount

Risk

Rating

Status

The exact visible attributes should adapt to the uploaded input data where possible.

Place:

[ Queue ]

near the table.

Every table row must be clickable.

When a row is clicked:

Open the detailed view for that specific claim containing:

- Claim-specific Power BI dashboard

- Risk information

- Supporting evidence

- LLM explanation

- Human review controls

The selected row must determine the displayed information.

==================================================

11. CLAIM QUEUE

==================================================

When:

[ Queue ]

is clicked:

The original detailed batch table should disappear.

Replace it with:

INVESTIGATION QUEUE

Columns:

Claim ID

Provider ID

Risk

Rating

Status

Sort the queue by Risk Score in DESCENDING order.

Highest-risk cases must appear first.

Maintain access to the overall batch dashboard.

==================================================

12. UNCOVER PROVIDERS PAGE

==================================================

Title:

UNCOVER PROVIDERS

Create two major action cards:

[ Single Provider ]

[ Batch Provider ]

Single Provider

→ Single Provider Analysis

Batch Provider

→ Batch Provider Analysis

==================================================

13. SINGLE PROVIDER ANALYSIS

==================================================

Title:

SINGLE PROVIDER ANALYSIS

Create approximately 10 realistic mock provider fields.

For example:

Provider ID / NPI

Provider Claim Count

Provider Beneficiary Count

Beneficiary Count

Reimbursed Amount

Deductible Amount

Days Admitted

Payment Per Beneficiary

Peer Deviation

Utilization Indicator

Include suitable checkbox/toggle inputs where appropriate.

Button:

[ Analyse Provider ]

After clicking:

Display:

1. Provider-specific Power BI Dashboard

2. LLM Explainability Report

3. Human Review controls

==================================================

14. SINGLE PROVIDER LLM EXPLANATION

==================================================

Create a polished report:

AI INVESTIGATION EXPLANATION

Display:

Provider Risk Score

Risk Rating

Key Signals

Peer Comparison

Behavioral Patterns

Supporting Evidence

Assessment

LLM Reasoning

The report should explain why the provider appears:

Potentially Suspicious

or

No Significant Anomaly Detected

Do not present the LLM as independently proving fraud.

Below the report:

[ Accept ]

[ Reject ]

Add:

AI-assisted assessment — Final decision requires domain-expert review.

==================================================

15. SINGLE PROVIDER DASHBOARD

==================================================

Display provider-specific analytics:

Risk Score

Risk Rating

Provider Claim Count

Beneficiary Count

Payment Statistics

Peer Deviation

Utilization

Behavioral Indicators

Temporal Indicators

Use attractive charts and metric cards.

==================================================

16. BATCH PROVIDER ANALYSIS

==================================================

Title:

BATCH PROVIDER ANALYSIS

Provide:

[ Upload Providers File ]

After upload:

Show file information.

Then reveal:

[ Analyse Provider ]

When clicked:

Display:

1. Overall Provider Dashboard

2. Provider Results Table

3. Investigate action

==================================================

17. BATCH PROVIDER DASHBOARD

==================================================

Display KPI cards:

Total Providers

Suspicious Cases

High Risk Providers

Medium Risk Providers

Low Risk Providers

Unique Beneficiaries

Total Claims

Average Risk Score

Add professional charts for:

Risk Distribution

Provider Distribution

Payment Distribution

Utilization Distribution

This dashboard represents the complete uploaded provider dataset.

==================================================

18. BATCH PROVIDER RESULTS TABLE

==================================================

Display provider attributes from the uploaded file.

Important columns:

Provider ID

Risk

Rating

Status

Also display relevant provider attributes.

Every row must be clickable.

When a row is clicked:

Open the selected provider's:

- Power BI Dashboard

- Risk details

- Supporting evidence

- LLM reasoning

- Human review controls

Add:

[ Investigate ]

==================================================

19. PROVIDER QUEUE

==================================================

When:

[ Investigate ]

is clicked:

The original provider results table disappears.

Display:

PROVIDER INVESTIGATION QUEUE

Columns:

Provider ID

Risk

Rating

Status

Sort by Risk Score in DESCENDING order.

Highest-risk providers appear first.

Keep the overall provider dashboard accessible.

==================================================

20. COMMON BATCH EXPERIENCE

==================================================

The Claim and Provider batch experiences must follow the same interaction pattern:

UPLOAD

 ↓

ANALYSE

 ↓

OVERALL DASHBOARD

 ↓

DETAILED TABLE

 ↓

CLICK ANY ROW

 ↓

INDIVIDUAL DASHBOARD + LLM EXPLANATION

 ↓

HUMAN REVIEW

Then:

QUEUE / INVESTIGATE

 ↓

PRIORITIZED RISK QUEUE

The user must be able to move between the overall dashboard, queue, and individual case details without losing context.

==================================================

21. POWER BI INTEGRATION

==================================================

IMPORTANT:

Embed Power BI dashboards directly inside the React frontend.

Power BI must appear as an integrated part of the application, not as an external page.

Use Power BI embedding for:

CLAIM:

- Single Claim Dashboard

- Batch Claim Dashboard

- Individual selected claim dashboard

PROVIDER:

- Single Provider Dashboard

- Batch Provider Dashboard

- Individual selected provider dashboard

The selected claim/provider should determine the relevant dashboard context where the existing Power BI setup supports filtering.

For batch analysis:

Display the overall Power BI dashboard for the uploaded dataset.

For row-level analysis:

Display the corresponding claim/provider Power BI dashboard alongside the LLM explanation.

The Power BI area should look native to the application using a polished dashboard card/container.

==================================================

22. COMMON HUMAN REVIEW

==================================================

Both Claim and Provider workflows must support human review.

Display:

HUMAN REVIEW

[ Accept ]

[ Reject ]

Clearly communicate:

The AI system provides anomaly detection and explainability.

The domain expert performs the final verification.

Use professional status indicators:

Pending Review

Accepted

Rejected

Resubmission Required

==================================================

23. VISUAL DESIGN SYSTEM

==================================================

Use a consistent design system throughout the entire application.

Components should feel unified:

- Floating cards

- Premium buttons

- KPI cards

- Dashboard cards

- Data tables

- Risk badges

- Status badges

- Explainability panels

- Review panels

- Upload cards

- Power BI containers

Buttons should have:

- subtle hover elevation

- smooth transitions

- clear active states

- professional typography

Tables should have:

- clean headers

- row hover states

- clear sorting

- readable spacing

- sticky headers where appropriate

==================================================

24. ANIMATIONS

==================================================

Use subtle high-quality animations:

Landing:

- Floating hero

- Fade-in title

- Button entrance

Navigation:

- Smooth page transitions

Cards:

- Hover elevation

- Slight movement

Dashboard:

- KPI entrance animation

- Smooth chart rendering

Tables:

- Row hover

- Detail transition

LLM report:

- Smooth reveal

Power BI:

- Smooth loading state

Do not make the application feel slow.

==================================================

25. RESPONSIVE DESIGN

==================================================

The application must work well on:

Desktop

Laptop

Tablet

The main investigation experience should prioritize desktop screens.

For smaller screens:

- Stack dashboard panels.

- Make tables horizontally scrollable.

- Keep important actions visible.

- Preserve readability.

==================================================

26. FINAL EXPERIENCE

==================================================

The complete user journey should feel like:

FWA RISK INVESTIGATOR

        ↓

ENTER SECURE PORTAL

        ↓

LOGIN

        ↓

UNCOVER FWA

        ↓

 ┌─────────────────┐

 │                 │

 ▼                 ▼

CLAIMS          PROVIDERS

 │                 │

 ▼                 ▼

SINGLE/BATCH    SINGLE/BATCH

 │                 │

 ▼                 ▼

ML ANALYSIS     ML ANALYSIS

 │                 │

 ▼                 ▼

ANALYTICAL      ANALYTICAL

ENGINE          ENGINE

 │                 │

 ▼                 ▼

RISK +          RISK +

EVIDENCE        EVIDENCE

 │                 │

 ▼                 ▼

LLM EXPLANATION + POWER BI

 │                 │

 ▼                 ▼

HUMAN REVIEW

 │

 ├── ACCEPT

 ├── REJECT

 └── RESUBMISSION

The final frontend should look like a real enterprise FWA payment-integrity investigation platform, not a generic CRUD dashboard.

Prioritize:

1. Visual quality

2. Clear navigation

3. Explainability

4. Power BI integration

5. Human review

6. Batch investigation workflow

7. Row-level investigation

8. Professional animations

9. Responsive design

10. Clean, maintainable UI

```

This project was built with [Lovable](https://lovable.dev).

**Live app**: https://fwa-analyzer-suite.lovable.app

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/63fae29d-6f07-4281-8657-1bc60a42c8a9).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
