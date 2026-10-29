"""Convert the public Multi-News validation split; run this script with python -I.

Download val.src.cleaned and val.tgt from the data links in
https://github.com/Alex-Fabbri/Multi-News and provide their paths below.
No dataset text is distributed with this code.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--documents',type=Path,required=True,help='Raw validation source, one cluster per line')
    p.add_argument('--summaries',type=Path,required=True,help='Raw validation target, one summary per line')
    p.add_argument('--output',type=Path,default=Path('data/multinews_validation.json'))
    args=p.parse_args()
    documents=args.documents.read_text(encoding='utf-8').splitlines()
    summaries=args.summaries.read_text(encoding='utf-8').splitlines()
    if len(documents)!=len(summaries) or not documents:
        p.error('The aligned validation files must have the same nonzero number of lines')
    # Preserve literal NEWLINE_CHAR tokens, matching the original experiment loader.
    rows=[]
    for idx,(document,summary) in enumerate(zip(documents,summaries)):
        articles=[a.strip() for a in document.split('|||||') if a.strip()]
        if not articles:p.error('Empty source cluster at index '+str(idx))
        rows.append(dict(articles=articles,summary=summary.strip(),idx=idx))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Wrote',len(rows),'examples to',args.output)


if __name__=='__main__':main()
