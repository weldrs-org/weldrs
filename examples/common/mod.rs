use polars::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;

const FIRST_NAMES: &[&str] = &[
    "James",
    "Mary",
    "Robert",
    "Patricia",
    "John",
    "Jennifer",
    "Michael",
    "Linda",
    "David",
    "Elizabeth",
    "William",
    "Barbara",
    "Richard",
    "Susan",
    "Joseph",
    "Jessica",
    "Thomas",
    "Sarah",
    "Charles",
    "Karen",
    "Christopher",
    "Lisa",
    "Daniel",
    "Betty",
    "Matthew",
    "Sandra",
    "Anthony",
    "Margaret",
    "Mark",
    "Ashley",
    "Donald",
    "Dorothy",
    "Steven",
    "Kimberly",
    "Paul",
    "Emily",
    "Andrew",
    "Donna",
    "Joshua",
    "Michelle",
    "Kenneth",
    "Carol",
    "Kevin",
    "Amanda",
    "Brian",
    "Melissa",
    "George",
    "Deborah",
    "Timothy",
    "Stephanie",
];

const SURNAMES: &[&str] = &[
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Perez",
    "Thompson",
    "White",
    "Harris",
    "Sanchez",
    "Clark",
    "Ramirez",
    "Lewis",
    "Robinson",
    "Walker",
    "Young",
    "Allen",
    "King",
    "Wright",
    "Scott",
    "Torres",
    "Nguyen",
    "Hill",
    "Flores",
    "Green",
    "Adams",
    "Nelson",
    "Baker",
    "Hall",
    "Rivera",
    "Campbell",
    "Mitchell",
    "Carter",
    "Roberts",
];

const CITIES: &[&str] = &[
    "London",
    "Manchester",
    "Birmingham",
    "Leeds",
    "Glasgow",
    "Liverpool",
    "Bristol",
    "Sheffield",
    "Edinburgh",
    "Cardiff",
    "Belfast",
    "Nottingham",
    "Newcastle",
    "Brighton",
    "Leicester",
    "Portsmouth",
    "Plymouth",
    "Oxford",
    "Cambridge",
    "York",
    "Bath",
    "Chester",
    "Exeter",
    "Norwich",
    "Derby",
    "Coventry",
    "Swansea",
    "Aberdeen",
    "Dundee",
    "Reading",
];

const EMAIL_DOMAINS: &[&str] = &[
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
    "protonmail.com",
    "mail.com",
    "aol.com",
    "zoho.com",
    "fastmail.com",
];

/// Perturb a string by swapping adjacent chars, dropping a char, or substituting a char.
fn perturb_string(s: &str, rng: &mut impl Rng) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= 1 {
        return s.to_string();
    }

    match rng.gen_range(0..3) {
        // Swap two adjacent characters
        0 => {
            let mut result = chars.clone();
            let idx = rng.gen_range(0..chars.len() - 1);
            result.swap(idx, idx + 1);
            result.into_iter().collect()
        }
        // Drop a character
        1 => {
            let idx = rng.gen_range(0..chars.len());
            chars
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, c)| *c)
                .collect()
        }
        // Substitute a character
        _ => {
            let mut result = chars;
            let idx = rng.gen_range(0..result.len());
            let replacement = (b'a' + rng.gen_range(0..26)) as char;
            result[idx] = replacement;
            result.into_iter().collect()
        }
    }
}

/// Generate a synthetic person dataset for record linkage testing.
///
/// Creates `n_unique` base records by sampling from name/city/email pools,
/// then generates duplicates (with perturbations) at the given `dup_rate`.
///
/// Returns a DataFrame with columns: `unique_id` (i64), `first_name`, `surname`,
/// `city`, `email` (nullable).
pub fn generate_person_dataset(n_unique: usize, dup_rate: f64, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut ids: Vec<i64> = Vec::new();
    let mut first_names: Vec<String> = Vec::new();
    let mut surnames: Vec<String> = Vec::new();
    let mut cities: Vec<String> = Vec::new();
    let mut emails: Vec<Option<String>> = Vec::new();

    let mut next_id: i64 = 1;

    for _ in 0..n_unique {
        let first = FIRST_NAMES[rng.gen_range(0..FIRST_NAMES.len())].to_string();
        let sur = SURNAMES[rng.gen_range(0..SURNAMES.len())].to_string();
        let city = CITIES[rng.gen_range(0..CITIES.len())].to_string();
        let domain = EMAIL_DOMAINS[rng.gen_range(0..EMAIL_DOMAINS.len())];
        let email = format!("{}.{}@{domain}", first.to_lowercase(), sur.to_lowercase());

        // Add base record
        ids.push(next_id);
        first_names.push(first.clone());
        surnames.push(sur.clone());
        cities.push(city.clone());
        emails.push(Some(email.clone()));
        next_id += 1;

        // Possibly create duplicates
        if rng.gen_bool(dup_rate.min(1.0)) {
            let n_dups = if rng.gen_bool(0.3) { 2 } else { 1 };
            for _ in 0..n_dups {
                ids.push(next_id);
                next_id += 1;

                // Perturb first name
                first_names.push(perturb_string(&first, &mut rng));

                // Surname: usually exact, rare typo (~10%)
                if rng.gen_bool(0.1) {
                    surnames.push(perturb_string(&sur, &mut rng));
                } else {
                    surnames.push(sur.clone());
                }

                // City: perturb
                cities.push(perturb_string(&city, &mut rng));

                // Email: different domain, None, or slight typo
                match rng.gen_range(0..3) {
                    0 => {
                        // Different domain
                        let new_domain = EMAIL_DOMAINS[rng.gen_range(0..EMAIL_DOMAINS.len())];
                        emails.push(Some(format!(
                            "{}.{}@{new_domain}",
                            first.to_lowercase(),
                            sur.to_lowercase()
                        )));
                    }
                    1 => {
                        // Missing email
                        emails.push(None);
                    }
                    _ => {
                        // Typo in email
                        emails.push(Some(perturb_string(&email, &mut rng)));
                    }
                }
            }
        }
    }

    let email_series = Series::new(
        PlSmallStr::from("email"),
        emails
            .iter()
            .map(|e| e.as_deref())
            .collect::<Vec<Option<&str>>>(),
    );

    DataFrame::new(vec![
        Column::new(PlSmallStr::from("unique_id"), &ids),
        Column::new(PlSmallStr::from("first_name"), &first_names),
        Column::new(PlSmallStr::from("surname"), &surnames),
        Column::new(PlSmallStr::from("city"), &cities),
        email_series.into(),
    ])
    .unwrap()
}

/// Print a summary of a DataFrame: label, row count, and first 5 rows.
#[allow(dead_code)]
pub fn print_df_summary(df: &DataFrame, label: &str) {
    println!("\n--- {label} ---");
    println!("Rows: {}", df.height());
    println!("{}", df.head(Some(5)));
}
