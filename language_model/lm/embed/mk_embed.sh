python mk_embed.py semantic_10.txt semantic_10_s.txt
python mk_embed.py ../../data/finance/subgoogle.txt google_s.txt

python imputation.py ../../data/finance/aff.txt ../../data/finance/subgoogle.txt 5 google_aff.txt
python mk_embed.py google_aff.txt google_aff_s.txt
rm google_aff.txt

python imputation.py ../../data/finance/aff.txt semantic_10.txt 5 semantic_10_aff.txt
python mk_embed.py semantic_10_aff.txt semantic_10_aff_s.txt
rm semantic_10_aff.txt

python imputation.py semantic_10.txt ../../data/finance/subgoogle.txt 5 google_semantic_10.txt
python mk_embed.py google_semantic_10.txt google_semantic_10_s.txt
rm google_semantic_10.txt

python imputation.py ../../data/finance/subgoogle.txt semantic_10.txt 5 semantic_10_google.txt
python mk_embed.py semantic_10_google.txt semantic_10_google_s.txt
rm semantic_10_google.txt


python mk_embed.py ../../data/finance/subglove.txt glove_s.txt
python mk_embed.py ../../data/finance/subfast.txt fast_s.txt

python imputation.py ../../data/finance/aff.txt ../../data/finance/subglove.txt 5 glove_aff.txt
python mk_embed.py glove_aff.txt glove_aff_s.txt
rm glove_aff.txt

python imputation.py ../../data/finance/aff.txt ../../data/finance/subfast.txt 5 fast_aff.txt
python mk_embed.py fast_aff.txt fast_aff_s.txt
rm fast_aff.txt

python imputation.py ../../data/finance/subglove.txt semantic_10.txt 5 semantic_10_glove.txt
python mk_embed.py semantic_10_glove.txt semantic_10_glove_s.txt
rm semantic_10_glove.txt

python imputation.py ../../data/finance/subfast.txt semantic_10.txt 5 semantic_10_fast.txt
python mk_embed.py semantic_10_fast.txt semantic_10_fast_s.txt
rm semantic_10_fast.txt


