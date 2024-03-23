import sacrebleu


def calc_bleu(src_jp: list, tgt_en: list):
    """
    BLEUスコアを計算する関数.

    :param src_jp: 翻訳元の日本語文. [sentence1, sentence2, ...]
    :param tgt_en: 機械翻訳による英訳. [translated1, translated2, ...]
    :return:
    """
    # BLEU スコアの計算
    bleu = sacrebleu.corpus_bleu(tgt_en, [src_jp])
    print(f"BLEU score: {bleu.score}")

    return bleu.score
