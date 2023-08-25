dict_code_UFR = {
    "UFR:014": "APR",
    "UFR:028": "ABC",
    "UFR:095": "AVC",
    "UFR:005": "BJN",
    "UFR:009": "BRK",
    "UFR:010": "BCT",
    "UFR:011": "BCH",
    "UFR:033": "BRT",
    "UFR:016": "BRC",
    "UFR:042": "CFX",
    "UFR:019": "CRC",
    "UFR:021": "CCH",
    "UFR:022": "CCL",
    "UFR:029": "ERX",
    "UFR:036": "GCL",
    "UFR:075": "EGC",
    "UFR:038": "HND",
    "UFR:026": "HMN",
    "UFR:099": "HAD",
    "UFR:041": "HTD",
    "UFR:032": "JVR",
    "UFR:044": "JFR",
    "UFR:047": "LRB",
    "UFR:049": "LRG",
    "UFR:053": "LMR",
    "UFR:061": "NCK",
    "UFR:096": "PBR",
    "UFR:066": "PSL",
    "UFR:068": "RPC",
    "UFR:069": "RMB",
    "UFR:070": "RDB",
    "UFR:072": "RTH",
    "UFR:073": "SAT",
    "UFR:079": "SPR",
    "UFR:076": "SLS",
    "UFR:084": "SSL",
    "UFR:087": "TNN",
    "UFR:088": "TRS",
    "UFR:090": "VGR",
    "UFR:064": "VPD",
    "UFR:INC": "INCONNU",
}


dict_hopital = {
    "APR": "UFR:014",
    "ABC": "UFR:028",
    "AVC": "UFR:095",
    "BJN": "UFR:005",
    "BRK": "UFR:009",
    "BCT": "UFR:010",
    "BCH": "UFR:011",
    "BRT": "UFR:033",
    "BRC": "UFR:016",
    "CFX": "UFR:042",
    "CRC": "UFR:019",
    "CCH": "UFR:021",
    "CCL": "UFR:022",
    "ERX": "UFR:029",
    "GCL": "UFR:036",
    "EGC": "UFR:075",
    "HND": "UFR:038",
    "HMN": "UFR:026",
    "HAD": "UFR:099",
    "HTD": "UFR:041",
    "JVR": "UFR:032",
    "JFR": "UFR:044",
    "LRB": "UFR:047",
    "LRG": "UFR:049",
    "LMR": "UFR:053",
    "NCK": "UFR:061",
    "PBR": "UFR:096",
    "PSL": "UFR:066",
    "RPC": "UFR:068",
    "RMB": "UFR:069",
    "RDB": "UFR:070",
    "RTH": "UFR:072",
    "SAT": "UFR:073",
    "SPR": "UFR:079",
    "SLS": "UFR:076",
    "SSL": "UFR:084",
    "TNN": "UFR:087",
    "TRS": "UFR:088",
    "VGR": "UFR:090",
    "VPD": "UFR:064",
}


def tables(schema, table):

    table_dict = {
        "cse": {
            "visits": "i2b2_visit",
            "visit_detail": "i2b2_observation_ufr",
            "documents": "i2b2_observation_doc",
            "patients": "i2b2_patient",
            "icd10": "i2b2_observation_cim10",
            "ccam": "i2b2_observation_ccam",
            "ghm": "i2b2_observation_ghm",
            "concept": "i2b2_concept",
        },
    }

    return table_dict["cse"][table]


age_partition = [7, 17, 25, 65, 120]

snippet_window_length = 35

unknown_and_other_forms = [
    "SA",
    "Suicide attempt",
    "Autolysis",
    "Self destructive behavior",
]
regex_sa = {
    "Suicide attempt": [
        r"(?i)tentative[s]?\s+de\s+sui?cide",
        r"(?i)tent[ée]\s+de\s+((se\s+(suicider|tuer))|(mettre\s+fin\s+[àa]\s+((ses\s+jours?)|(sa\s+vie))))",
    ],
    "SA": [
        r"\b(?<!\.)(?<!Voie\s\d\s\:\s)(?<!Voie\sd.abord\s\:\s)(?<!surface\s)(?<!d[ée]sorientation\s)(?<!abord\s)(?<!ECG\s:\s)(?<!volume\s)(?<!\d\s[mc]m\sde\sla\s)(?<!\d[mc]m\sde\sla\s)(?<!au\scontact\sde\sla\s)T\.?S\.?(?![\.A-Za-z])(?!\sapyr[eé]tique)(?!.+TRANSSEPTAL)(?!.+T[34])(?!.+en\sr.gression)\b",
        r"(?<!\.)T\.S\.(?![A-Za-z])",
        r"\b(?<!.)TS\.\B",
    ],
    "Autolysis": [r"(?i)tentative\s+d'autolyse", r"(?i)autolyse"],
    "Intentional drug overdose": [
        r"(?i)(intoxication|ingestion)\s+m[ée]dicamenteuse\s+volontaire",
        r"(?i)\b(i\.?m\.?v\.?)\b",
        r"(?i)(intoxication|ingestion)\s*([a-zA-Z0-9_éàèôê\-]+\s*){0,3}\s*volontaire",
        r"TS\s+med\s+polymedicamenteuse",
        r"TS\s+(poly)?([\s-])?m[ée]dicamenteuse",
    ],
    "Jumping from height": [
        r"(?i)tentative[s]?\s+de\s+d[ée]fenestration",
        r"(?i)(?<!id[ée]e\sde\s)d[ée]fenestration(?!\saccidentelle)",
        r"(?i)d[ée]fenestration\s+volontaire",
        r"(?i)d[ée]fenestration\s+intentionnelle",
        r"(?i)jet.r?\sd.un\spont",
    ],
    "Cuts": [r"(?i)phl[ée]botomie"],
    "Strangling": [r"(?i)pendaison"],
    "Self destructive behavior": [r"(?i)autodestruction"],
    "Burn/gas/caustic": [r"(?i)ingestion\sde\s(produit\s)?caustique"],
}

regex_rf = {
    "sexual_violence": [
        r"(?i)((agressions?)|(attaques?)|(atteintes?)|violences?|menaces?|outrages?)\s+(sexuel(le)?s?|sexistes?)",
        r"(?i)abus\s+sexuel",
        r"(?i)victime\s+d'abus",
        r"(?i)p[éeè]dophilie",
        r"(?i)\bviols?\b",
        r"(?i)attouchement",
        r"(?i)harc[èeé]lements?\s+sexuel",
        r"(?i)p[èeé]n[èeé]tration",
        r"(?i)p[èeé]n[èeé]tr[èeé]r",
        r"(?i)attentat\s+[àa]\s+la\s+pudeur",
        r"(?i)s[èeé]vice\s+sexuel",
        r"(?i)\babus[èeé]\b",
        r"(?i)cyber\s+agression\s+[àa]\s+caract[éeè]re\s+sexuel(le)?",
        r"(?i)((mutilation)|(traumatisme)|(l[éeè]sions?))\s+((g[éeè]nitale?)|(hym[éeè]n[éeè]ale))",
        r"(?i)(propos|comportements?)\s+[aà]\s+connotation\s+(sexuel(le)|sexiste)?",
        r"(?i)rapport\ssexuel\s((non\sconsenti)|forcé|(de\sforce))",
        r"(?i)gestes?\sd[eé]placés?",
    ],
    "domestic_violence": [
        r"(?i)((tensions?)|(agressions?)|(violences?)|(conflits?)|(harc[èeé]lements?)|(brimades?)|(injures?)|tortures?)\s*((familiale?)|(dans\s+la\s+famille)|(((au)|(dans\s+le))\s+foyer)|(domestique)|(conjugale?)|(à\s+la\s+maison)|((par|avec)\s+(son|sa)\s+conjointe?)|(intrafamiliaux)|(intrafamiliale?))",
        r"(?i)conjugopathie",
    ],
    "physical_violence": [
        r"(?i)(violences?|maltraitances?|s.vices?|agressions?|attaques?)\s+physiques?",
        r"(?i)r45\.?6",
        r"(?i)t74\.?1",
        r"(?i)\bbattue?\b",
    ],
    "social_isolation": [
        r"(?i)(isolement\ssocial)|(solitude)|(solitaire)|(mal\s+du\s+pays)|(z60\.?2)",
        r"(?i)(sentir\s+seule?)",
        r"(?i)((sentir|patiente?|enfant)\s+isolée?)",
        r"(?i)(sent\s+seule?)",
        r"(?i)pas\sde\sfamille",
        r"(?i)pas\sd.animal\sdomestique",
        r"(?i)isolement\sdans\sla\sclasse",
    ],
}


covid_date = "2020-03-01"

# Type I error
alpha = 0.05

# Labels/constants for plot
sex_label = {"M": "Male", "W": "Female"}

filled_marker_style = dict(
    linestyle="-",
    markersize=10,
    markeredgecolor="black",
)


# Type of hospital reference (15 hospitals selection)
hospital_type_dict = {
    "ABC": "Adult & Paediatric",
    "APR": "Adult & Paediatric",
    "BCH": "Adult",
    "BCT": "Adult & Paediatric",
    "HMN": "Adult",
    "JVR": "Adult & Paediatric",
    "LMR": "Adult & Paediatric",
    "LRB": "Adult",
    "NCK": "Adult & Paediatric",
    "PBR": "Adult",
    "PSL": "Adult",
    "SAT": "Adult",
    "SLS": "Adult",
    "TNN": "Adult",
    "TRS": "Paediatric",
}


seconds_per_day = 60 * 60 * 24
