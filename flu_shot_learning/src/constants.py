BINARY_FEATURES_KEYS = ['behavioral_antiviral_meds',
                        'behavioral_avoidance',
                        'behavioral_face_mask',
                        'behavioral_wash_hands',
                        'behavioral_large_gatherings',
                        'behavioral_outside_home',
                        'behavioral_touch_face',
                        'doctor_recc_h1n1',
                        'doctor_recc_seasonal',
                        'chronic_med_condition',
                        'child_under_6_months',
                        'health_worker',
                        'health_insurance',
                        # other binary features
                        'sex',
                        'marital_status',
                        'rent_or_own'
                        ]
                    

ORDERNAL_FEATURES_KEYS = ['opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                          'opinion_seas_vacc_effective', 'opinion_seas_risk',
                          'opinion_seas_sick_from_vacc']
                          

OTHER_FEATURES_KEYS = ['age_group',
                       'education',
                       'race',
                       'income_poverty',
                       'employment_status',
                       'hhs_geo_region',
                       'census_msa',
                       'household_adults',
                       'household_children',
                       'employment_industry',
                       'employment_occupation'
                       ]

NUMERICAL_FEATURES = ['opinion_h1n1_risk',
                      'opinion_seas_vacc_effective',
                      'household_children',
                      'h1n1_concern',
                      'household_adults',
                      'opinion_seas_risk',
                      'opinion_h1n1_vacc_effective',
                      'opinion_h1n1_sick_from_vacc',
                      'h1n1_knowledge',
                      'opinion_seas_sick_from_vacc']


CATEGORICAL_FEATURES = ['race',
                        #'employment_industry',
                        'behavioral_outside_home',
                        'census_msa',
                        'health_insurance',
                        'income_poverty',
                        'employment_status',
                        'behavioral_touch_face',
                        'behavioral_face_mask',
                        'behavioral_wash_hands',
                        'child_under_6_months',
                        #'employment_occupation',
                        'sex',
                        'behavioral_antiviral_meds',
                        'doctor_recc_h1n1',
                        'doctor_recc_seasonal',
                        'health_worker',
                        'marital_status',
                        #'hhs_geo_region',
                        'rent_or_own',
                        'age_group',
                        'education',
                        'behavioral_avoidance',
                        'behavioral_large_gatherings',
                        'chronic_med_condition']


ALL_FEATURES = []
ALL_FEATURES.extend(NUMERICAL_FEATURES)
ALL_FEATURES.extend(CATEGORICAL_FEATURES)
