import csv

# Mapping of original headers to descriptive names
header_mapping = {
    "label": "crop_type_class",
    "f1": "sigHH_Rad05July", "f2": "sigHV_Rad05July", "f3": "sigVV_Rad05July", "f4": "sigRR_Rad05July", "f5": "sigRL_Rad05July",
    "f6": "sigLL_Rad05July", "f7": "Rhhvv_Rad05July", "f8": "Rhvhh_Rad05July", "f9": "Rhvvv_Rad05July", "f10": "Rrrll_Rad05July",
    "f11": "Rrlrr_Rad05July", "f12": "Rrlll_Rad05July", "f13": "Rhh_Rad05July", "f14": "Rhv_Rad05July", "f15": "Rvv_Rad05July",
    "f16": "Rrr_Rad05July", "f17": "Rrl_Rad05July", "f18": "Rll_Rad05July", "f19": "Ro12_Rad05July", "f20": "Ro13_Rad05July",
    "f21": "Ro23_Rad05July", "f22": "Ro12cir_Rad05July", "f23": "Ro13cir_Rad05July", "f24": "Ro23cir_Rad05July", "f25": "l1_Rad05July",
    "f26": "l2_Rad05July", "f27": "l3_Rad05July", "f28": "H_Rad05July", "f29": "A_Rad05July", "f30": "a_Rad05July",
    "f31": "HA_Rad05July", "f32": "H1mA_Rad05July", "f33": "1mHA_Rad05July", "f34": "1mH1mA_Rad05July", "f35": "PH_Rad05July",
    "f36": "rvi_Rad05July", "f37": "paulalpha_Rad05July", "f38": "paulbeta_Rad05July", "f39": "paulgamma_Rad05July", "f40": "krogks_Rad05July",
    "f41": "krogkd_Rad05July", "f42": "krogkh_Rad05July", "f43": "freeodd_Rad05July", "f44": "freedbl_Rad05July", "f45": "freevol_Rad05July",
    "f46": "yamodd_Rad05July", "f47": "yamdbl_Rad05July", "f48": "yamhlx_Rad05July", "f49": "yamvol_Rad05July", "f50": "sigHH_Rad14July",
    # ... (Mapping continues for all 174 features)
}

# Note: For brevity in the display, I've truncated the dictionary. 
# The script below will dynamically handle the headers provided in your text.

def convert_with_new_headers(input_file, output_file):
    # Your provided list of names in order
    new_names = [
        "crop_type_class", "sigHH_Rad05July", "sigHV_Rad05July", "sigVV_Rad05July", "sigRR_Rad05July", "sigRL_Rad05July", "sigLL_Rad05July", "Rhhvv_Rad05July", "Rhvhh_Rad05July", "Rhvvv_Rad05July", "Rrrll_Rad05July", "Rrlrr_Rad05July", "Rrlll_Rad05July", "Rhh_Rad05July", "Rhv_Rad05July", "Rvv_Rad05July", "Rrr_Rad05July", "Rrl_Rad05July", "Rll_Rad05July", "Ro12_Rad05July", "Ro13_Rad05July", "Ro23_Rad05July", "Ro12cir_Rad05July", "Ro13cir_Rad05July", "Ro23cir_Rad05July", "l1_Rad05July", "l2_Rad05July", "l3_Rad05July", "H_Rad05July", "A_Rad05July", "a_Rad05July", "HA_Rad05July", "H1mA_Rad05July", "1mHA_Rad05July", "1mH1mA_Rad05July", "PH_Rad05July", "rvi_Rad05July", "paulalpha_Rad05July", "paulbeta_Rad05July", "paulgamma_Rad05July", "krogks_Rad05July", "krogkd_Rad05July", "krogkh_Rad05July", "freeodd_Rad05July", "freedbl_Rad05July", "freevol_Rad05July", "yamodd_Rad05July", "yamdbl_Rad05July", "yamhlx_Rad05July", "yamvol_Rad05July", "sigHH_Rad14July", "sigHV_Rad14July", "sigVV_Rad14July", "sigRR_Rad14July", "sigRL_Rad14July", "sigLL_Rad14July", "Rhhvv_Rad14July", "Rhvhh_Rad14July", "Rhvvv_Rad14July", "Rrrll_Rad14July", "Rrlrr_Rad14July", "Rrlll_Rad14July", "Rhh_Rad14July", "Rhv_Rad14July", "Rvv_Rad14July", "Rrr_Rad14July", "Rrl_Rad14July", "Rll_Rad14July", "Ro12_Rad14July", "Ro13_Rad14July", "Ro23_Rad14July", "Ro12cir_Rad14July", "Ro13cir_Rad14July", "Ro23cir_Rad14July", "l1_Rad14July", "l2_Rad14July", "l3_Rad14July", "H_Rad14July", "A_Rad14July", "a_Rad14July", "HA_Rad14July", "H1mA_Rad14July", "1mHA_Rad14July", "1mH1mA_Rad14July", "PH_Rad14July", "rvi_Rad14July", "paulalpha_Rad14July", "paulbeta_Rad14July", "paulgamma_Rad14July", "krogks_Rad14July", "krogkd_Rad14July", "krogkh_Rad14July", "freeodd_Rad14July", "freedbl_Rad14July", "freevol_Rad14July", "yamodd_Rad14July", "yamdbl_Rad14July", "yamhlx_Rad14July", "yamvol_Rad14July", "B_Opt05July", "G_Opt05July", "R_Opt05July", "Redge_Opt05July", "NIR_Opt05July", "NDVI_Opt05July", "SR_Opt05July", "RGRI_Opt05July", "EVI_Opt05July", "ARVI_Opt05July", "SAVI_Opt05July", "NDGI_Opt05July", "gNDVI_Opt05July", "MTVI2_Opt05July", "NDVIre_Opt05July", "SRre_Opt05July", "NDGIre_Opt05July", "RTVIcore_Opt05July", "RNDVI_Opt05July", "TCARI_Opt05July", "TVI_Opt05July", "PRI2_Opt05July", "MeanPC1_Opt05July", "VarPC1_Opt05July", "HomPC1_Opt05July", "ConPC1_Opt05July", "DisPC1_Opt05July", "EntPC1_Opt05July", "SecMomPC1_Opt05July", "CorPC1_Opt05July", "MeanPC2_Opt05July", "VarPC2_Opt05July", "HomPC2_Opt05July", "ConPC2_Opt05July", "DisPC2_Opt05July", "EntPC2_Opt05July", "SecMomPC2_Opt05July", "CorPC2_Opt05July", "B_Opt14July", "G_Opt14July", "R_Opt14July", "Redge_Opt14July", "NIR_Opt14July", "NDVI_Opt14July", "SR_Opt14July", "RGRI_Opt14July", "EVI_Opt14July", "ARVI_Opt14July", "SAVI_Opt14July", "NDGI_Opt14July", "gNDVI_Opt14July", "MTVI2_Opt14July", "NDVIre_Opt14July", "SRre_Opt14July", "NDGIre_Opt14July", "RTVIcore_Opt14July", "RNDVI_Opt14July", "TCARI_Opt14July", "TVI_Opt14July", "PRI2_Opt14July", "MeanPC1_Opt14July", "VarPC1_Opt14July", "HomPC1_Opt14July", "ConPC1_Opt14July", "DisPC1_Opt14July", "EntPC1_Opt14July", "SecMomPC1_Opt14July", "CorPC1_Opt14July", "MeanPC2_Opt14July", "VarPC2_Opt14July", "HomPC2_Opt14July", "ConPC2_Opt14July", "DisPC2_Opt14July", "EntPC2_Opt14July", "SecMomPC2_Opt14July", "CorPC2_Opt14July"
    ]

    try:
        with open(input_file, 'r') as f_in:
            # Skip the first line (the old labels) and read the rest
            lines = f_in.readlines()
            data_lines = lines[1:] 

        with open(output_file, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            # Write the NEW descriptive header
            writer.writerow(new_names)
            
            # Write the data rows
            for line in data_lines:
                if line.strip():
                    writer.writerow(line.strip().split(','))
                    
        print(f"File '{output_file}' created successfully with descriptive headers.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function
convert_with_new_headers('WinnipegDataset.txt','crop_mapping.csv')