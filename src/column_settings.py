total_builds_new = ['all_types_total_buildings' ] 
                   

res = [ 
 'all_types_premise_area_total',
 'all_types_total_fl_area_H_total',
 'all_types_total_fl_area_FC_total',
 'all_types_uprn_count_total',
 'mixed_alltypes_count',
 'comm_alltypes_count',
 'unknown_alltypes_count',
 'clean_res_total_buildings',
 'clean_res_premise_area_total',
 'clean_res_total_fl_area_H_total',
 'clean_res_total_fl_area_FC_total',
 'clean_res_base_floor_total',
 'clean_res_basement_heated_vol_total',
 'clean_res_listed_bool_total',
#  'derived_unknown_res',
 'all_res_total_buildings',
 'percent_residential',
  'all_res_total_fl_area_H_total',
 'all_res_total_fl_area_FC_total',
 'confidence_floor_area',
'clean_res_uprn_count_total'
]

new_all_var_res = [
 'clean_res_total_fl_area_H_total',
 'all_res_total_buildings',
'clean_res_uprn_count_total']

extra_res = ['clean_res_uprn_count_total', ]
minimial_res = ['all_types_total_buildings', 'all_types_uprn_count_total', 'clean_res_total_buildings', 'all_res_total_buildings']


outb = [
 'outb_res_total_buildings',
 'outb_res_premise_area_total',
 'outb_res_total_fl_area_H_total',
 'outb_res_total_fl_area_FC_total',
 ]

type_cols = [
'2 storeys terraces with t rear extension_pct',
'3-4 storey and smaller flats_pct',
'Domestic outbuilding_pct',
'Large detached_pct',
'Large semi detached_pct',
'Linked and step linked premises_pct',
'Medium height flats 5-6 storeys_pct',
'Planned balanced mixed estates_pct',
'Semi type house in multiples_pct',
'Small low terraces_pct',
'Standard size detached_pct',
'Standard size semi detached_pct',
'Tall flats 6-15 storeys_pct',
'Tall terraces 3-4 storeys_pct',
'Very large detached_pct',
'Very tall point block flats_pct',
'all_unknown_typology',
]


type_cols2 = [
'2 storeys terraces with t rear extension_pct',
'3-4 storey and smaller flats_pct',
'Domestic outbuilding_pct',
'Large detached_pct',
'Large semi detached_pct',
'Linked and step linked premises_pct',
'Medium height flats 5-6 storeys_pct',
'Planned balanced mixed estates_pct',
'Semi type house in multiples_pct',
'Small low terraces_pct',
'Standard size detached_pct',
'Standard size semi detached_pct',
'Tall flats 6-15 storeys_pct',
'Tall terraces 3-4 storeys_pct',
'Very large detached_pct',
'Very tall point block flats_pct',
'all_unknown_typology_pct',
]

age_cols = [
'1919-1944_pct',
'1945-1959_pct',
'1960-1979_pct',
'1980-1989_pct',
'1990-1999_pct',
'Post 1999_pct',
'Pre 1919_pct',
 ] 

temp_cols = ['HDD',
'CDD',
'HDD_summer',
'CDD_summer',
'HDD_winter',
] 

postcode_geoms = [
'postcode_area',
'postcode_density',
'log_pc_area',
]

pc_area  = ['postcode_area']

region_cols =[ 
'region',
'oa21cd',
 'lsoa21cd',
 'msoa21cd',
 'ladcd'
]

energy_metrics = [ 'num_meters_gas',
 'total_gas',
 'avg_gas',
 'median_gas',
 'num_meters_elec',
 'total_elec',
 'avg_elec',
 'median_elec'
 ]
 

domain_invariant_inc_age = ['postcode_area',
'postcode_density',
'1919-1944_pct',
'1945-1959_pct',
'1960-1979_pct',
'1980-1989_pct',
'1990-1999_pct',
'all_none_age_pct',
'Post 1999_pct',
'Pre 1919_pct',
'all_res_total_buildings',
'all_types_premise_area_total',
'all_types_total_fl_area_H_total', 
] 
domain_invariant = ['postcode_area',
'postcode_density',
'all_types_premise_area_total',
'all_types_total_fl_area_H_total'
]



rural_urban = [
'RUC11CD',
'RUC11']

economic_census = [ 
# 'economic_activity_perc_Does not apply',
'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Part-time',
'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed with employees: Part-time',
'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed with employees: Full-time',
'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed without employees: Part-time',
'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed without employees: Full-time',
'economic_activity_perc_Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
'economic_activity_perc_Economically active and a full-time student: In employment: Employee: Part-time',
'economic_activity_perc_Economically active and a full-time student: In employment: Employee: Full-time',
'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed with employees: Part-time',
'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed with employees: Full-time',
'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed without employees: Part-time',
'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed without employees: Full-time',
'economic_activity_perc_Economically active and a full-time student: Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
'economic_activity_perc_Economically inactive: Retired',
'economic_activity_perc_Economically inactive: Student',
'economic_activity_perc_Economically inactive: Looking after home or family',
'economic_activity_perc_Economically inactive: Long-term sick or disabled',
'economic_activity_perc_Economically inactive: Other',
]

education_census = [
    #   'highest_qual_perc_Does not apply',
'highest_qual_perc_No qualifications',
'highest_qual_perc_Level 1 and entry level qualifications: 1 to 4 GCSEs grade A* to C, Any GCSEs at other grades, O levels or CSEs (any grades), 1 AS level, NVQ level 1, Foundation GNVQ, Basic or Essential Skills',
'highest_qual_perc_Level 2 qualifications: 5 or more GCSEs (A* to C or 9 to 4), O levels (passes), CSEs (grade 1), School Certification, 1 A level, 2 to 3 AS levels, VCEs, Intermediate or Higher Diploma, Welsh Baccalaureate Intermediate Diploma, NVQ level 2, Intermediate GNVQ, City and Guilds Craft, BTEC First or General Diploma, RSA Diploma',
'highest_qual_perc_Apprenticeship',
'highest_qual_perc_Level 3 qualifications: 2 or more A levels or VCEs, 4 or more AS levels, Higher School Certificate, Progression or Advanced Diploma, Welsh Baccalaureate Advance Diploma, NVQ level 3; Advanced GNVQ, City and Guilds Advanced Craft, ONC, OND, BTEC National, RSA Advanced Diploma',
'highest_qual_perc_Level 4 qualifications or above: degree (BA, BSc), higher degree (MA, PhD, PGCE), NVQ level 4 to 5, HNC, HND, RSA Higher Diploma, BTEC Higher level, professional qualifications (for example, teaching, nursing, accountancy)',
'highest_qual_perc_Other: vocational or work-related qualifications, other qualifications achieved in England or Wales, qualifications achieved outside England or Wales (equivalent not stated or unknown)',
]

occupation_census = [
    #  'occupation_perc_Does not apply',
'occupation_perc_1. Managers, directors and senior officials',
'occupation_perc_2. Professional occupations',
'occupation_perc_3. Associate professional and technical occupations',
'occupation_perc_4. Administrative and secretarial occupations',
'occupation_perc_5. Skilled trades occupations',
'occupation_perc_6. Caring, leisure and other service occupations',
'occupation_perc_7. Sales and customer service occupations',
'occupation_perc_8. Process, plant and machine operatives',
'occupation_perc_9. Elementary occupations',
]

ethnic_census = [ 
# 'ethnic_group_perc_Does not apply',
'ethnic_group_perc_Asian, Asian British or Asian Welsh: Bangladeshi',
'ethnic_group_perc_Asian, Asian British or Asian Welsh: Chinese',
'ethnic_group_perc_Asian, Asian British or Asian Welsh: Indian',
'ethnic_group_perc_Asian, Asian British or Asian Welsh: Pakistani',
'ethnic_group_perc_Asian, Asian British or Asian Welsh: Other Asian',
'ethnic_group_perc_Black, Black British, Black Welsh, Caribbean or African: African',
'ethnic_group_perc_Black, Black British, Black Welsh, Caribbean or African: Caribbean',
'ethnic_group_perc_Black, Black British, Black Welsh, Caribbean or African: Other Black',
'ethnic_group_perc_Mixed or Multiple ethnic groups: White and Asian',
'ethnic_group_perc_Mixed or Multiple ethnic groups: White and Black African',
'ethnic_group_perc_Mixed or Multiple ethnic groups: White and Black Caribbean',
'ethnic_group_perc_Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups',
'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
'ethnic_group_perc_White: Irish',
'ethnic_group_perc_White: Gypsy or Irish Traveller',
'ethnic_group_perc_White: Roma',
'ethnic_group_perc_White: Other White',
'ethnic_group_perc_Other ethnic group: Arab',
'ethnic_group_perc_Other ethnic group: Any other ethnic group',
]

socio_class_census = [ 
    # 'socio_class_perc_Does not apply',
 'socio_class_perc_L1, L2 and L3: Higher managerial, administrative and professional occupations',
 'socio_class_perc_L4, L5 and L6: Lower managerial, administrative and professional occupations',
 'socio_class_perc_L7: Intermediate occupations',
 'socio_class_perc_L8 and L9: Small employers and own account workers',
 'socio_class_perc_L10 and L11: Lower supervisory and technical occupations',
 'socio_class_perc_L12: Semi-routine occupations',
 'socio_class_perc_L13: Routine occupations',
 'socio_class_perc_L14.1 and L14.2: Never worked and long-term unemployed',
 'socio_class_perc_L15: Full-time students',
 ]

household_size_census = [ 
'household_siz_perc_perc_0 people in household',
'household_siz_perc_perc_1 person in household',
'household_siz_perc_perc_2 people in household',
'household_siz_perc_perc_3 people in household',
'household_siz_perc_perc_4 people in household',
'household_siz_perc_perc_5 people in household',
'household_siz_perc_perc_6 people in household',
'household_siz_perc_perc_7 people in household',
'household_siz_perc_perc_8 or more people in household',

]
occupancy_census = [
    #   'occupancy_rating_perc_Does not apply',
'occupancy_rating_perc_Occupancy rating of bedrooms: +2 or more',
'occupancy_rating_perc_Occupancy rating of bedrooms: +1',
'occupancy_rating_perc_Occupancy rating of bedrooms: 0',
'occupancy_rating_perc_Occupancy rating of bedrooms: -1',
'occupancy_rating_perc_Occupancy rating of bedrooms: -2 or less',
]


household_comp_census= [
    # 'household_comp_perc_Does not apply',
 'household_comp_perc_One-person household',
 'household_comp_perc_Single family household: All aged 66 years and over',
 'household_comp_perc_Single family household: Couple family household',
 'household_comp_perc_Single family household: Lone parent household',
 'household_comp_perc_Other household types']

tenure_census = [ 
    #  'tenure_perc_Does not apply',
 'tenure_perc_Owned: Owns outright',
 'tenure_perc_Owned: Owns with a mortgage or loan',
 'tenure_perc_Shared ownership: Shared ownership',
 'tenure_perc_Social rented: Rents from council or Local Authority',
 'tenure_perc_Social rented: Other social rented',
 'tenure_perc_Private rented: Private landlord or letting agency',
 'tenure_perc_Private rented: Other private rented',
 'tenure_perc_Lives rent free',
]


sex_census = [ 'sex_perc_Female',
'sex_perc_Male'
]

central_heat_census = [ 
# 'central_heating_perc_Does not apply',
'central_heating_perc_No central heating',
'central_heating_perc_Mains gas only',
'central_heating_perc_Tank or bottled gas only',
'central_heating_perc_Electric only',
'central_heating_perc_Wood only',
'central_heating_perc_Solid fuel only',
'central_heating_perc_Renewable energy only',
'central_heating_perc_District or communal heat networks only',
'central_heating_perc_Other central heating only',
'central_heating_perc_Two or more types of central heating (not including renewable energy)',
'central_heating_perc_Two or more types of central heating (including renewable energy)'
]


deprivation = [
    #  'deprivation_perc_Does not apply',
'deprivation_perc_Household is not deprived in any dimension',
'deprivation_perc_Household is deprived in one dimension',
'deprivation_perc_Household is deprived in two dimensions',
'deprivation_perc_Household is deprived in three dimensions',
'deprivation_perc_Household is deprived in four dimensions',
]

bedrooms_census = [ 
    # 'bedroom_number_perc_Does not apply',
 'bedroom_number_perc_1 bedroom',
 'bedroom_number_perc_2 bedrooms',
 'bedroom_number_perc_3 bedrooms',
 'bedroom_number_perc_4 or more bedrooms',
]

all_census = economic_census + education_census  + occupation_census + ethnic_census  + household_size_census + occupancy_census + household_comp_census + bedrooms_census+ tenure_census + deprivation + sex_census + socio_class_census + central_heat_census
all_vars = total_builds_new + res + outb + type_cols + age_cols + temp_cols + postcode_geoms + region_cols + all_census + rural_urban
all_vars_excl_census = total_builds_new + res + outb + type_cols + age_cols + temp_cols + postcode_geoms + region_cols  

all_vars_energy = energy_metrics +  minimial_res + outb + type_cols + age_cols + temp_cols + postcode_geoms + region_cols + all_census + rural_urban

ndvi_cols = ['max_ndvi']

new_all_vars =  new_all_var_res+  outb + type_cols + age_cols + temp_cols + postcode_geoms + region_cols + all_census + rural_urban

new_all_vars2 = new_all_var_res+  outb + type_cols + age_cols + temp_cols + postcode_geoms + all_census 
new_all_vars3 =  new_all_var_res+  outb + type_cols + age_cols + temp_cols + postcode_geoms  + all_census + rural_urban


predi_eui = type_cols + age_cols + temp_cols + postcode_geoms  + all_census + rural_urban

sobol_eth = ['deprivation_perc_Household is deprived in two dimensions',
'deprivation_perc_Household is deprived in three dimensions',
'deprivation_perc_Household is deprived in four dimensions',
 'tenure_perc_Social rented: Rents from council or Local Authority',
 'tenure_perc_Social rented: Other social rented',
 'tenure_perc_Private rented: Private landlord or letting agency',
 'tenure_perc_Private rented: Other private rented',
 'economic_activity_perc_Economically inactive: Retired',
'economic_activity_perc_Economically inactive: Student',
'economic_activity_perc_Economically inactive: Looking after home or family',
'economic_activity_perc_Economically inactive: Long-term sick or disabled',
'economic_activity_perc_Economically inactive: Other',
'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
'ethnic_group_perc_White: Irish',
'ethnic_group_perc_White: Gypsy or Irish Traveller',
'ethnic_group_perc_White: Roma',
'ethnic_group_perc_White: Other White',
 ]

sool_eth2= ['deprivation_perc_Household is deprived in two dimensions',
'deprivation_perc_Household is deprived in three dimensions',
'deprivation_perc_Household is deprived in four dimensions',
 'tenure_perc_Social rented: Rents from council or Local Authority',
 'tenure_perc_Social rented: Other social rented',
 'tenure_perc_Private rented: Private landlord or letting agency',
 'tenure_perc_Private rented: Other private rented',
 'economic_activity_perc_Economically inactive: Retired',
'economic_activity_perc_Economically inactive: Student',
'economic_activity_perc_Economically inactive: Looking after home or family',
'economic_activity_perc_Economically inactive: Long-term sick or disabled',
'economic_activity_perc_Economically inactive: Other',
'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
 
 ]

gasperhhcols = type_cols + age_cols + temp_cols + postcode_geoms + new_all_var_res + rural_urban + sool_eth2
gasperhhcols2 = type_cols + age_cols + temp_cols + postcode_geoms + new_all_var_res + rural_urban + all_census + region_cols + outb

settings_col_dict_census = {
    0: ['Economic', economic_census],
    1: ['Education', education_census],
    2: ['Ethnicity', ethnic_census],
    3: ['Rural Urban', rural_urban],
    4: ['Household Size', household_size_census],
    5: ['Occupancy', occupancy_census],
    6: ['Household Comp', household_comp_census],
    7: ['SocioEcon Classification', socio_class_census],
    8: ['Central Heat', central_heat_census],
    9: ['Deprivation', deprivation],
    10: ['Occupation', occupation_census],
    11: ['Sex', sex_census],
    12: ["Households size", household_size_census], 
    13: ['Tenure' , tenure_census],
    14: ['Bedrooms', bedrooms_census],
    15: ['All Census', all_census],
} 


feat_cols= ['all_types_total_fl_area_H_total',
 'all_types_premise_area_total',
  'clean_res_total_buildings',
 'clean_res_premise_area_total',
 'clean_res_total_fl_area_H_total',
 'clean_res_base_floor_total',
 'Domestic outbuilding_pct',
 'Standard size detached_pct',
 'Standard size semi detached_pct',
 'Small low terraces_pct',
 '2 storeys terraces with t rear extension_pct',
 'Pre 1919_pct',
 'all_none_age_pct',
 '1960-1979_pct',
 '1919-1944_pct',
 'Post 1999_pct',
 'HDD',
 'CDD',
 'HDD_summer',
 'HDD_winter',
 'postcode_area',
 'postcode_density',
 'log_pc_area',
 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
 'central_heating_perc_Mains gas only',
 'household_siz_perc_perc_1 person in household']

feat_cols_new = ['all_types_total_buildings', 
'all_res_total_fl_area_H_total', 
'all_types_total_fl_area_H_total', 
'all_types_total_fl_area_FC_total', 
'all_types_premise_area_total', 
'outb_res_total_fl_area_H_total',
'outb_res_total_buildings',
'Domestic outbuilding_pct',
'Standard size detached_pct',
'Standard size semi detached_pct',
'1960-1979_pct',    
'1919-1944_pct',
'1945-1959_pct',
'CDD',
'HDD_winter',
'HDD_summer',
'postcode_density',
'postcode_area',
'oa21cd',
'msoa21cd',
'household_siz_perc_perc_1 person in household',
'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
'socio_class_perc_L1, L2 and L3: Higher managerial, administrative and professional occupations',
 'socio_class_perc_L4, L5 and L6: Lower managerial, administrative and professional occupations',
 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
 'household_comp_perc_One-person household',
 'household_comp_perc_Single family household: All aged 66 years and over',
]

feat_cols_final = ['all_types_uprn_count_total',
'ladcd',
'clean_res_uprn_count_total',
'all_res_total_fl_area_H_total',
'Pre 1919_pct',
'all_types_total_fl_area_H_total',
'clean_res_premise_area_total',
'all_types_total_fl_area_FC_total',
'1919-1944_pct',
'all_types_premise_area_total',
'clean_res_total_buildings',
'all_res_total_fl_area_FC_total',
'clean_res_total_fl_area_H_total',
'msoa21cd',
'Standard size detached_pct',
'clean_res_total_fl_area_FC_total',
'region',
'postcode_density',
'Post 1999_pct', ] 

feat_cols_final_excl_region = ['all_types_uprn_count_total',
'clean_res_uprn_count_total',
'all_res_total_fl_area_H_total',
'Pre 1919_pct',
'all_types_total_fl_area_H_total',
'clean_res_premise_area_total',
'all_types_total_fl_area_FC_total',
'1919-1944_pct',
'all_types_premise_area_total',
'clean_res_total_buildings',
'all_res_total_fl_area_FC_total',
'clean_res_total_fl_area_H_total',
'Standard size detached_pct',
'clean_res_total_fl_area_FC_total',
'postcode_density',
'Post 1999_pct' ] 

FC_3 = ['all_types_uprn_count_total',
'clean_res_uprn_count_total',
'all_res_total_fl_area_H_total',
'Pre 1919_pct',
'all_types_total_fl_area_H_total',
'clean_res_premise_area_total',
'all_types_total_fl_area_FC_total',
'1919-1944_pct',
'all_types_premise_area_total',
'clean_res_total_buildings',
'all_res_total_fl_area_FC_total',
'clean_res_total_fl_area_H_total',
'Standard size detached_pct',
'clean_res_total_fl_area_FC_total',
'postcode_density',
'Post 1999_pct', 
'household_siz_perc_perc_1 person in household',
'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
'socio_class_perc_L1, L2 and L3: Higher managerial, administrative and professional occupations',
 'socio_class_perc_L4, L5 and L6: Lower managerial, administrative and professional occupations',
 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
 'household_comp_perc_One-person household',
 'household_comp_perc_Single family household: All aged 66 years and over',
 ] 

FC_25 = ['all_types_uprn_count_total',
 'clean_res_uprn_count_total',
 'ladcd',
 'all_res_total_fl_area_H_total',
 'Pre 1919_pct',
 'clean_res_total_buildings',
 'clean_res_premise_area_total',
 '1919-1944_pct',
 'msoa21cd',
 'all_types_premise_area_total',
 'Standard size detached_pct',
 'postcode_density',
 'clean_res_total_fl_area_H_total',
 'region',
 'postcode_area',
 'Post 1999_pct',
 'household_siz_perc_perc_1 person in household',
'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
'socio_class_perc_L1, L2 and L3: Higher managerial, administrative and professional occupations',
 'socio_class_perc_L4, L5 and L6: Lower managerial, administrative and professional occupations',
 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
 'household_comp_perc_One-person household',
 'household_comp_perc_Single family household: All aged 66 years and over']

FC25_noregion = ['all_types_uprn_count_total',
 'clean_res_uprn_count_total',
 'all_res_total_fl_area_H_total',
 'Pre 1919_pct',
 'clean_res_total_buildings',
 'clean_res_premise_area_total',
 '1919-1944_pct',
 'all_types_premise_area_total',
 'Standard size detached_pct',
 'postcode_density',
 'clean_res_total_fl_area_H_total',
 'postcode_area',
 'Post 1999_pct',
 'household_siz_perc_perc_1 person in household',
'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
'socio_class_perc_L1, L2 and L3: Higher managerial, administrative and professional occupations',
 'socio_class_perc_L4, L5 and L6: Lower managerial, administrative and professional occupations',
 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
 'household_comp_perc_One-person household',
 'household_comp_perc_Single family household: All aged 66 years and over']


fc_sensible = ['all_types_uprn_count_total',
 'clean_res_uprn_count_total',
 'all_res_total_fl_area_H_total',
 'Pre 1919_pct',
 'clean_res_total_buildings',
 'clean_res_premise_area_total',
 '1919-1944_pct',
 'all_types_premise_area_total',
 'Standard size detached_pct',
 'postcode_density',
 'clean_res_total_fl_area_H_total',
 'postcode_area',
 'Post 1999_pct',
 'household_siz_perc_perc_1 person in household',
'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
'socio_class_perc_L1, L2 and L3: Higher managerial, administrative and professional occupations',
 'socio_class_perc_L4, L5 and L6: Lower managerial, administrative and professional occupations',
 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
 'household_comp_perc_One-person household',
 'household_comp_perc_Single family household: All aged 66 years and over']


fc_final = [ 
'all_res_total_fl_area_H_total',
'Pre 1919_pct',
'Standard size detached_pct',
'postcode_area',
'HDD_winter',  
'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
 'socio_class_perc_L1, L2 and L3: Higher managerial, administrative and professional occupations',
 'household_comp_perc_One-person household',
 'Domestic outbuilding_pct',
 '3-4 storey and smaller flats_pct',
 ] 


fc_final_added = [ 
'all_res_total_fl_area_H_total',
'Pre 1919_pct',
'Standard size detached_pct',
'postcode_area',
'HDD_winter',  
'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
 'socio_class_perc_L1, L2 and L3: Higher managerial, administrative and professional occupations',
 'household_comp_perc_One-person household',
 'tenure_perc_Owned: Owns outright',
 'bedroom_number_perc_3 bedrooms',
 'Domestic outbuilding_pct',
 '3-4 storey and smaller flats_pct',
 'central_heating_perc_Mains gas only',
 'tenure_perc_Private rented: Private landlord or letting agency', 
 'bedroom_number_perc_2 bedrooms',
 ] 
 

sobol1 = ['all_types_total_buildings',
 'clean_res_total_buildings',
 'clean_res_premise_area_total',
 'clean_res_total_fl_area_H_total',
 'clean_res_uprn_count_total',
 'outb_res_total_buildings',
 'outb_res_premise_area_total',
 '2 storeys terraces with t rear extension_pct',
 '3-4 storey and smaller flats_pct',
 'Domestic outbuilding_pct',
 'Large detached_pct',
 'Large semi detached_pct',
 'Linked and step linked premises_pct',
 'Medium height flats 5-6 storeys_pct',
 'Planned balanced mixed estates_pct',
 'Semi type house in multiples_pct',
 'Small low terraces_pct',
 'Standard size detached_pct',
 'Standard size semi detached_pct',
 'Tall flats 6-15 storeys_pct',
 'Tall terraces 3-4 storeys_pct',
 'Very large detached_pct',
 'Very tall point block flats_pct',
 '1919-1944_pct',
 '1945-1959_pct',
 '1960-1979_pct',
 '1980-1989_pct',
 '1990-1999_pct',
 'Post 1999_pct',
 'Pre 1919_pct',
 'CDD_summer',
 'HDD_winter',
 'postcode_area',
 'postcode_density',
 ]


 






fc_final_2 = [ 
'all_res_total_fl_area_H_total',
'Pre 1919_pct',
'Standard size detached_pct',
'postcode_density',
'HDD_winter',  
'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
 'socio_class_perc_L1, L2 and L3: Higher managerial, administrative and professional occupations',
 'household_comp_perc_One-person household',
 'Domestic outbuilding_pct',
 '3-4 storey and smaller flats_pct',
 'region'
 ] 
single_floor_area  = ['all_res_total_fl_area_H_total']

mixed_use_vars = new_all_var_res + type_cols + age_cols+ temp_cols + ['postcode_area', 'postcode_density'] + all_census 
mixed_use_vars2 = new_all_var_res + type_cols2 + age_cols+ temp_cols + ['postcode_area', 'postcode_density'] + all_census 
mix_use_excl_census = new_all_var_res + type_cols + age_cols+ temp_cols + ['postcode_area', 'postcode_density'] +  sobol_eth

norm_except_fl_area = type_cols + age_cols + all_census + postcode_geoms + temp_cols + rural_urban
minimal_norm_except_fl_are = all_census + postcode_geoms + temp_cols + rural_urban


mixed_excl_cols = ['CDD_summer',
 'HDD_winter',
 'household_siz_perc_perc_0 people in household',
 'household_comp_perc_One-person household',
 'sex_perc_Male',
 'socio_class_perc_L15: Full-time students']

mixed_cola_nocorr = [x for x in mixed_use_vars if x not in mixed_excl_cols ]
mixed_no_census =  [ x for x in mixed_cola_nocorr if x not in all_census ]
mixednocorr_minc_censu = mixed_no_census+ sool_eth2


mixed_cols2_nocrr = [x for x in mixed_use_vars2 if x not in mixed_excl_cols ]
mixed_no_census2 =  [ x for x in mixed_cols2_nocrr if x not in all_census ]
mixednocorr_minc_censu2 = mixed_no_census2 + sool_eth2

 

settings_dict = {
0: ['COB' , total_builds_new ] , 
1: ['Residential BS', res],
2: ['Outbuildings BS', outb],
3: ['Typology', type_cols],
4: ['Age', age_cols],
5: ['Temperature', temp_cols],
6: ['Local Morph.', postcode_geoms],
7: ['Region', region_cols],
9: ['Socio-Demogs', all_census ] ,
10: ['Urban/Rural',  rural_urban],
11 : ['COB, Res BS', res  + total_builds_new],
12: ['COB, Type', total_builds_new + type_cols ],
13: ['COB, Region', total_builds_new + region_cols ], 
14: ['COB, Age', total_builds_new + age_cols ],
15: ['COB, Temp', total_builds_new + temp_cols ],
16: ['COB, Local Morph.', total_builds_new + postcode_geoms ],
17: ['COB, Urban/Rural', total_builds_new  + rural_urban],
18: ['All vars', all_vars], 
19: ['COB, Region, PC Geom, Temp', total_builds_new + region_cols  +postcode_geoms +temp_cols ],
20: ['NDVI', ndvi_cols], 
21: ['COB, NDVI' , total_builds_new + ndvi_cols],
22: ['COB, NDVI, Temp, Urban/Rural' , total_builds_new + ndvi_cols + temp_cols + rural_urban], 
23: ['COB, NDVI, Temp, Urban/Rural, Local Morph.' , total_builds_new + ndvi_cols + temp_cols + rural_urban + postcode_geoms ], 
24: ['COB, NDVI, Temp, Urban/Rural, Local Morph., Type' ,  total_builds_new + ndvi_cols + temp_cols + rural_urban + postcode_geoms + type_cols ], 
25: ['COB, NDVI, Temp, Urban/Rural, Local Morph., Socio-Demogs' , total_builds_new + ndvi_cols + temp_cols + rural_urban + postcode_geoms + all_census ], 
26: ['COB, NDVI, Temp, Urban/Rural, Socio-Demogs' , total_builds_new + ndvi_cols + temp_cols + rural_urban  + all_census ], 
27: ['COB, NDVI, Temp,\n Socio-Demogs' , total_builds_new + ndvi_cols + temp_cols  + all_census ], 
28: ['COB, NDVI, Local Morph.\n Temp, Socio-Demogs' , total_builds_new + ndvi_cols + temp_cols  + all_census + postcode_geoms ], 
29: ['COB, Temp,\n Socio-Demogs', total_builds_new + temp_cols + all_census ],
30: ['COB, Temp, \n Socio-Demogs, Local Morph.' , total_builds_new + temp_cols  + all_census + postcode_geoms ], 
31: ['COB, Temp,\n NDVI', total_builds_new + temp_cols + ndvi_cols ],
32: ['COB, Temp, \n NDVI, Local Morph.' , total_builds_new + temp_cols  + ndvi_cols + postcode_geoms ], 
33: ['Temp, \n NDVI, Local Morph.' ,   temp_cols  + ndvi_cols + postcode_geoms ], 
34: ['Temp, Socio-Demogs \n NDVI, Local Morph.' ,  all_census +  temp_cols  + ndvi_cols + postcode_geoms ], 
35: ['Temp', 'Local Morph', temp_cols + postcode_geoms ] , 
36: ['Temp, Socio-Demogs, \n  Local Morph.' ,  all_census +  temp_cols + postcode_geoms ], 
37: ['Temp + PC area', pc_area + temp_cols] , 
38: ['PC area', pc_area ] ,
39: ['All Vars excl. Census', all_vars_excl_census] ,
40: ['Temp, NDVI', temp_cols+ ndvi_cols], 
41: ['Temp, NDVI + Urban/Rural', temp_cols+ ndvi_cols+ rural_urban], 
42: ['Domain Invariant (inc Age)' , domain_invariant_inc_age],
43: ['Domain Invariant' , domain_invariant],
44: ['Feature Imp Cols', feat_cols],
45: ['Feature Imp Cols New', feat_cols_new],
46: ['Feature Imp Cols Final', feat_cols_final],
47: ['Feature Imp Cols Final Excl. Region', feat_cols_final_excl_region],
48: ['Feat Imp 3',FC_3 ], 
49: ['Feat Imp 25', FC_25], 
50: ['FI + region' , feat_cols + ['oa21cd']] , 
51: ['Feat Imp 25 no region',  FC25_noregion ],
52: ['FI_final_minimal', fc_final + ['region'] ] , 
53: ['Single_fl' , single_floor_area] , 
54: ['test_FI_density', fc_final_2 ], 
55: ['FI_final_minimal_added', fc_final_added + ['region'] ] ,
56: ['all_vars_inc_energy', all_vars_energy],
57: ['all_res_andhouseholds_nongasPerc', all_vars_excl_census + ['perc_non_gas', 'count_households'] ],
58: ['All cols excl BS', norm_except_fl_area ], 
59: ['All cols excl BS + energy', norm_except_fl_area + energy_metrics  ],
60: ['All cols excl BS + energy + household size', norm_except_fl_area + energy_metrics +['count_of_households'] ],
61: ['All cols excl BS + energy + household size+nongasperc', norm_except_fl_area + energy_metrics +['count_of_households', 'perc_non_gas'] ],
62: ['All cols excl BS + energy + household size+nongasperc + FL Are', norm_except_fl_area + energy_metrics +['count_of_households', 'perc_non_gas', 'clean_res_total_fl_area_H_total'] ],
63: ['Most cols excl BS  + UPRN', norm_except_fl_area + extra_res  + ['count_of_households', 'perc_non_gas'] ],
64: ['Most cols excl BS', norm_except_fl_area   + ['count_of_households', 'perc_non_gas'] ],
65: ['Min cols excl all BS  + UPRN ', minimal_norm_except_fl_are  + extra_res  + ['count_of_households', 'perc_non_gas'] ],
66: ['Most cols excl BS  + UPRN excl nongas HS', norm_except_fl_area + extra_res   ],
67: ['gea', ['residential_GEA_total'] ] , 
68: ['gia', ['residential_GIA_total'] ] , 
69: ['gea + gia' , ['residential_GEA_total', 'residential_GIA_total'] ] ,
70: ['sobol1', sobol1],
71: ['sobol2', sobol1  +['ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',  'household_siz_perc_perc_1 person in household' ] ],
72: ['sobol3', sobol1  + sobol_eth ],
74: ['new all vars', new_all_vars], 
75: ['new all vars2', new_all_vars2],
76: ['new all vars3', new_all_vars3],
77: ['mixed use vars', mixed_use_vars ] , 
78: ['mixed use cols with mimimal census', mix_use_excl_census ] , 
79: ['cols to predcit eui', predi_eui],
80: ['cols to predcit euicens', predi_eui + [ 'all_res_total_buildings', 'clean_res_uprn_count_total'] ], 
81: ['cols predcit gas per HH',  gasperhhcols],
82: ['cols predcit gas per HH2',  gasperhhcols2],
83: ['77 with corr removed' , mixed_cola_nocorr],
84: ['83 no census', mixed_no_census ], 
85: ['83 with extras', mixed_cola_nocorr + ['outb_res_premise_area_total', 'RUC11CD', 'region'] ],
86: ['83 and extras minus census', mixed_no_census + ['outb_res_premise_area_total', 'RUC11CD', 'region'] ],
87: ['83 with min ceneus', mixednocorr_minc_censu],
88: ['87 with pct unknt', mixednocorr_minc_censu2 ],
89: ['87 with no cnesus', mixed_no_census2 + ['all_unknown_typology']],
90: ['87 with no cnesus', mixed_no_census2 + ['all_unknown_typology']],
91: ['COB, NDVI, Temp, Urban/Rural, Local Morph., Socio-Demogs' , total_builds_new + ndvi_cols + temp_cols + rural_urban + postcode_geoms + sool_eth2 ], 
}  


