import requests
from lxml import etree
from tqdm import tqdm

from bs4 import BeautifulSoup

alphabet = {c for c in "ACDEFGHIKLMNPQRSTVWY"}

# https://rest.uniprot.org/uniprotkb/P37231.fasta
wildtype_sequence = """MGETLGDSPIDPESDSFTDTLSANISQEMTMVDTEMPFWPTNFGISSVDLSVMEDHSHSF
DIKPFTTVDFSSISTPHYEDIPFTRTDPVVADYKYDLKLQEYQSAIKVEPASPPYYSEKT
QLYNKPHEEPSNSLMAIECRVCGDKASGFHYGVHACEGCKGFFRRTIRLKLIYDRCDLNC
RIHKKSRNKCQYCRFQKCLAVGMSHNAIRFGRMPQAEKEKLLAEISSDIDQLNPESADLR
ALAKHLYDSYIKSFPLTKAKARAILTGKTTDKSPFVIYDMNSLMMGEDKIKFKHITPLQE
QSKEVAIRIFQGCQFRSVEAVQEITEYAKSIPGFVNLDLNDQVTLLKYGVHEIIYTMLAS
LMNKDGVLISEGQGFMTREFLKSLRKPFGDFMEPKFEFAVKFNALELDDSDLAIFIAVII
LSGDRPGLLNVKPIEDIQDNLLQALELQLKLNHPESSQLFAKLLQKMTDLRQIVTEHVQL
LQVIKKTETDMSLHPLLQEIYKDLY
""".replace("\n", "")

records = []
for i, from_aa in tqdm(list(enumerate(wildtype_sequence))):
    for to_aa in alphabet - {from_aa}:
        mutation = f"{from_aa}{i+1}{to_aa}"
        query = f"http://miter.broadinstitute.org/mitergrade/?query=p.{mutation}&prevalence=1.0e-5"
        r = requests.get(query)
        soup = BeautifulSoup(r.text, "html.parser")
        tree = etree.HTML(str(soup))
        records.append({
            "mutation": mutation,
            "experimental_function_score": float(tree.xpath("/html/body/div[2]/div/div[2]/div/div/div/div[1]/table/tbody/tr[4]/td[2]/text()")[0]),
            "fpld3_prediction": tree.xpath("/html/body/div[2]/div/div[2]/div/div/div/div[1]/table/tbody/tr[7]/td[2]/text()")[0],
            "type_2_diabetes_prediction": tree.xpath("/html/body/div[2]/div/div[2]/div/div/div/div[1]/table/tbody/tr[8]/td[2]/text()")[0]
        })

pl.from_records(records).write_csv("data/alphams_maves/dms_data/PPARG_HUMAN.csv")
