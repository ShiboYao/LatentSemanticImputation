cd ../rnn
source activate dl


for i in {1..10}
do
	python ptb_word_lm.py --data_path=../../data/finance 
done

for i in {1..10}
do
    python ptb_word_lm.py --data_path=../../data/finance --use_pre=../embed/semantic_10_s.txt
done

for i in {1..10}
do 
    python ptb_word_lm.py --data_path=../../data/finance --use_pre=../embed/semantic_10_aff_s.txt
done

for i in {1..10}
do
    python ptb_word_lm.py --data_path=../../data/finance --use_pre=../embed/semantic_10_google_s.txt
done

for i in {1..10}
do
    python ptb_word_lm.py --data_path=../../data/finance --use_pre=../embed/google_s.txt
done

for i in {1..10}
do
    python ptb_word_lm.py --data_path=../../data/finance --use_pre=../embed/google_aff_s.txt
done

for i in {1..10}
do
    python ptb_word_lm.py --data_path=../../data/finance --use_pre=../embed/glove_s.txt
done

for i in {1..10}
do
    python ptb_word_lm.py --data_path=../../data/finance --use_pre=../embed/glove_aff_s.txt
done

for i in {1..10}
do
    python ptb_word_lm.py --data_path=../../data/finance --use_pre=../embed/semantic_10_glove_s.txt
done


for i in {1..10}
do
    python ptb_word_lm.py --data_path=../../data/finance --use_pre=../embed/fast_s.txt
done


for i in {1..10}
do
    python ptb_word_lm.py --data_path=../../data/finance --use_pre=../embed/fast_aff_s.txt
done


for i in {1..10}
do
    python ptb_word_lm.py --data_path=../../data/finance --use_pre=../embed/semantic_10_fast_s.txt
done

