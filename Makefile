NER_INPUT = ./data/ner_data_raw/infered_tags.csv
TAGGER_TOKENS = ./data/tagger_data_raw/tokenized_texts_tagged.csv
TAGGER_TEXTS = ./data/tagger_data_raw/text_description.csv

.PHONY: all clean
all: ./data/colision_data/error_comptabilized.csv ./data/colision_data/detailled_collisions.csv ./data/merge_output/anomalies.csv

clean:
	rm -rf ./data/merge_output ./data/ner_data_processed ./data/tagger_data_processed ./data/colision_data/
	mkdir ./data/merge_output
	mkdir ./data/ner_data_processed
	mkdir ./data/tagger_data_processed
	mkdir ./data/colision_data

./data/ner_data_processed/sha_fixed.csv: $(NER_INPUT)
	uv run ./fix_sha.py $^ $@

./data/ner_data_processed/word_piece_resolved.csv: ./data/ner_data_processed/sha_fixed.csv
	uv run ./word_piece_merge_v2.py $^ $@

./data/tagger_data_processed/position_matched.csv: $(TAGGER_TEXTS) $(TAGGER_TOKENS)
	uv run ./position_matcher.py $^ $@

./data/tagger_data_processed/bilou_stripped.csv: ./data/tagger_data_processed/position_matched.csv
	uv run ./bilou_strip.py $^ $@

./data/merge_output/merged_stripped_v1.csv: ./data/ner_data_processed/word_piece_resolved.csv ./data/tagger_data_processed/bilou_stripped.csv
	uv run ./merge_data_src_v1.py $^ $@

./data/tagger_data_processed/tag_merged.csv: ./data/tagger_data_processed/position_matched.csv
	uv run ./word_piece_merge_v2.py $^ $@

./data/merge_output/merged_tag_merged_v1.csv: ./data/ner_data_processed/word_piece_resolved.csv ./data/tagger_data_processed/tag_merged.csv
	uv run ./merge_data_src_v1.py $^ $@

./data/merge_output/merged_tag_merged_v2.csv: ./data/ner_data_processed/word_piece_resolved.csv ./data/tagger_data_processed/tag_merged.csv
	uv run ./merge_data_src_v2.py $^ $@

./data/merge_output/merged_stripped_v2.csv: ./data/ner_data_processed/word_piece_resolved.csv ./data/tagger_data_processed/bilou_stripped.csv
	uv run ./merge_data_src_v2.py $^ $@

./data/colision_data/colision_list.csv: ./data/merge_output/merged_tag_merged_v2.csv
	uv run ./tag_match_analysis.py $^ $@

./data/colision_data/error_comptabilized.csv: ./data/colision_data/colision_list.csv
	uv run ./overlap_categorization.py $^ $@

./data/colision_data/detailled_collisions.csv: ./data/colision_data/colision_list.csv ./data/merge_output/merged_tag_merged_v2.csv $(TAGGER_TEXTS)
	uv run ./detail_collision_cmp.py $^ $@

./data/merge_output/anomalies.csv: ./data/merge_output/merged_tag_merged_v2.csv
	uv run ./extract_merge_anomalies.py $^ $@
