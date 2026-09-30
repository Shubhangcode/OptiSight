"""
recommendations.py
Evidence-based clinical recommendations for each Diabetic Retinopathy severity stage
based on AAO (American Academy of Ophthalmology) and ADA (American Diabetes Association) guidelines.
"""

from typing import TypedDict


class RecommendationResult(TypedDict):
    stage: int
    stage_label: str
    urgency: str
    urgency_color: str      # For UI color-coding
    action_summary: str
    recommendations: list[str]
    follow_up_interval: str
    icd10_code: str


CLINICAL_GUIDELINES: dict[int, RecommendationResult] = {
    0: RecommendationResult(
        stage=0,
        stage_label="No Diabetic Retinopathy",
        urgency="Low",
        urgency_color="#10b981",   # green
        action_summary="Routine annual diabetic eye screening recommended.",
        recommendations=[
            "Schedule comprehensive dilated fundus examination annually.",
            "Maintain strict glycemic control — target HbA1c < 7.0% (53 mmol/mol).",
            "Control blood pressure to < 130/80 mmHg per ADA guidelines.",
            "Manage lipid levels — target LDL < 100 mg/dL (or < 70 mg/dL if high cardiovascular risk).",
            "Advise patient on lifestyle modifications: low-glycemic diet, 150 min/week aerobic exercise.",
            "Educate patient on symptoms warranting urgent review: sudden vision change, floaters, flashes.",
            "Document absence of retinopathy for baseline monitoring.",
        ],
        follow_up_interval="12 months",
        icd10_code="E11.319",
    ),
    1: RecommendationResult(
        stage=1,
        stage_label="Mild Non-Proliferative DR",
        urgency="Low-Moderate",
        urgency_color="#eab308",   # yellow
        action_summary="Follow-up examination in 6–12 months; intensify metabolic risk factor control.",
        recommendations=[
            "Schedule follow-up dilated fundus examination in 6–12 months.",
            "Intensify glycemic control — target HbA1c < 7.0%; consider medication review with endocrinologist.",
            "Blood pressure monitoring — maintain < 130/80 mmHg; consider ACE inhibitors or ARBs if hypertensive and diabetic.",
            "Perform lipid profile assessment; initiate or optimize statin therapy if indicated.",
            "Reinforce importance of smoking cessation — smoking accelerates DR progression.",
            "Educate patient: microaneurysms are present but vision remains unaffected at this stage.",
            "Consider referral to diabetes care team for metabolic risk management.",
        ],
        follow_up_interval="6–12 months",
        icd10_code="E11.321",
    ),
    2: RecommendationResult(
        stage=2,
        stage_label="Moderate Non-Proliferative DR",
        urgency="Moderate",
        urgency_color="#f97316",   # orange
        action_summary="Ophthalmologist referral in 3–6 months; screen for Diabetic Macular Edema (DME).",
        recommendations=[
            "Refer to ophthalmologist or retinal specialist for comprehensive evaluation within 3–6 months.",
            "Obtain Optical Coherence Tomography (OCT) to screen for Diabetic Macular Edema (DME).",
            "If DME is confirmed: consider intravitreal anti-VEGF injections (e.g., ranibizumab, bevacizumab) or focal laser photocoagulation.",
            "Optimize HbA1c — target < 7.0%; consider insulin therapy or GLP-1 receptor agonists if not at target.",
            "Strict blood pressure management — target < 130/80 mmHg; review nephrology status.",
            "Perform renal function tests — microalbuminuria may indicate concurrent nephropathy.",
            "Advise patient to report any new visual symptoms immediately.",
            "Document fundus photography for longitudinal progression monitoring.",
        ],
        follow_up_interval="3–6 months",
        icd10_code="E11.3311",
    ),
    3: RecommendationResult(
        stage=3,
        stage_label="Severe Non-Proliferative DR",
        urgency="High",
        urgency_color="#ef4444",   # red
        action_summary="Urgent retina specialist evaluation within 2–4 weeks; assess for anti-VEGF or PRP.",
        recommendations=[
            "Urgent referral to retinal specialist — evaluation within 2–4 weeks.",
            "High risk of progression to Proliferative DR (PDR) within 1 year (~50% risk without intervention).",
            "Evaluate for panretinal photocoagulation (PRP) vs. intravitreal anti-VEGF based on OCT and FA findings.",
            "Obtain fluorescein angiography (FA) to assess ischemic areas and neovascular risk.",
            "OCT angiography (OCT-A) for non-invasive assessment of retinal vasculature.",
            "Aggressively optimize HbA1c, blood pressure, and lipid profile.",
            "Counsel patient on risk of imminent vision-threatening complications.",
            "Coordinate with endocrinologist, cardiologist, and nephrologist for systemic risk management.",
            "Avoid strenuous physical activity that may precipitate vitreous hemorrhage.",
        ],
        follow_up_interval="2–4 weeks",
        icd10_code="E11.3411",
    ),
    4: RecommendationResult(
        stage=4,
        stage_label="Proliferative Diabetic Retinopathy",
        urgency="Critical",
        urgency_color="#881337",   # dark red/crimson
        action_summary="Immediate specialized retina care required. Vision-threatening condition.",
        recommendations=[
            "IMMEDIATE referral to vitreoretinal surgeon — same-day or next-day evaluation.",
            "Panretinal photocoagulation (PRP) is the established treatment to ablate neovascularization.",
            "Intravitreal anti-VEGF therapy (ranibizumab, bevacizumab, or aflibercept) — may be used as adjunct or primary therapy.",
            "If vitreous hemorrhage or tractional retinal detachment is present: vitrectomy surgery may be indicated.",
            "OCT and fluorescein angiography urgently to map disease extent.",
            "Bilateral eye examination mandatory — PDR commonly affects both eyes.",
            "Strict systemic control — hospitalisation may be warranted for intensive glycemic stabilisation.",
            "Patient must be educated on risk of irreversible blindness without prompt treatment.",
            "Driving restrictions may apply — assess visual acuity and visual field status.",
            "Coordinate emergency care pathway with diabetes specialist team.",
        ],
        follow_up_interval="Immediate / Same day",
        icd10_code="E11.3511",
    ),
}


def get_clinical_recommendations(class_id: int) -> RecommendationResult:
    """
    Returns structured, evidence-based clinical recommendations for a given DR severity class.

    Args:
        class_id: Integer class ID (0–4) from model inference.

    Returns:
        RecommendationResult containing stage metadata and action plan.
    """
    if class_id not in CLINICAL_GUIDELINES:
        raise ValueError(f"Invalid class_id {class_id}. Must be between 0 and 4.")
    return CLINICAL_GUIDELINES[class_id]


def format_recommendation_for_api(class_id: int) -> dict:
    """
    Returns recommendation data formatted for JSON API response.
    """
    rec = get_clinical_recommendations(class_id)
    return {
        "stage": rec["stage"],
        "stage_label": rec["stage_label"],
        "urgency": rec["urgency"],
        "urgency_color": rec["urgency_color"],
        "action_summary": rec["action_summary"],
        "recommendations": rec["recommendations"],
        "follow_up_interval": rec["follow_up_interval"],
        "icd10_code": rec["icd10_code"],
    }


if __name__ == "__main__":
    print("Testing clinical recommendations module...\n")
    for cid in range(5):
        rec = format_recommendation_for_api(cid)
        print(f"Class {cid}: {rec['stage_label']}")
        print(f"  Urgency        : {rec['urgency']}")
        print(f"  Action Summary : {rec['action_summary']}")
        print(f"  Follow-up      : {rec['follow_up_interval']}")
        print(f"  ICD-10         : {rec['icd10_code']}")
        print(f"  Recommendations ({len(rec['recommendations'])} items):")
        for r in rec["recommendations"]:
            print(f"    - {r}")
        print()
    print("All recommendations generated successfully.")
