# RIPOSTE Risk Assessments

Scripts that complete a risk assessment for epidemiology case studies, specifically cholera in Cameroon and measles in DRC. See the details of the methodologies employed in the DRC feasibility study and the Cameroon risk analysis: https://rodekruis.sharepoint.com/sites/510-CRAVK-510/Gedeelde%20%20documenten/%5BPRJ%5D%20RIPOSTE/Final%20Deliverables?csf=1&web=1&e=DW4hli

This folder contains 4 types of scripts:
1. Data cleaning - datacleaning_cholera_Cameroon.py and datacleaning_measles_DRC.py
3. Adjusting for underreporting - adjust_under-reporting_CholeraCameroon.py and adjust_under-reporting_MeaslesDRC.py
4. Risk assessment - RiskAssessment_CholeraCameroon.py and RiskAssessment_MeaslesDRC.py
5. Precipitation analysis Cameroon - Precipitation_data_extraction_CMR.py & cases_rainfall.ipynb
6. Archive - a couple intermediary scripts that were a work in progress and have potentially interesting functions to refer back to

The data cleaning scripts read in all the necessary data for the risk assessment and merge them to a common spatial and temporal resolution. The data referred to in these scripts can be found here: https://rodekruis.sharepoint.com/sites/510-CRAVK-510/Gedeelde%20%20documenten/%5BPRJ%5D%20RIPOSTE/Cholera%20Cameroon/Trigger%20Model%20%26%20Risk%20Index/Data?csf=1&web=1&e=KTgcBr

The adjusting for underreporting scripts use distance from health care facilities to scale up the number of cases and deaths due to the concept of underreporting.

The risk assessment scripts normalize the data and apply a weighting model in order to agreggate the indicators into a risk score and create risk maps.

The precipitation scripts were used to extract daily precipitation data of Cameroon from the Copernicus Data Store and assess this data to determine the trigger threshold for the Cholera EAP in Cameroon.
