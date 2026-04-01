import re, random
from datasets import load_dataset

random.seed(42)


class CategoryOne:
    def __init__(self, data):
        self.data = data
        self.container = list()

    def search_data(self):
        c = 0
        for n, i in enumerate(self.data, start=1):
            x = re.search(
                r"\b(?:python|js|javascript|sql|html|css|c\+\+|java|code|debug|algorithm|function|programming|software|script)\b",
                i["instruction"], re.IGNORECASE)
            if x and "http" not in i["input"] and "http" not in i["output"] and "http" not in i[
                "instruction"] and "invoke" not in i['instruction'].lower():
                self.container.append(i)
                c += 1

    def final_filter(self, hash_container=False):
        S, M, L = list(), list(), list()
        for i in self.container:
            if 200 <= len(i["output"]) < 380:
                S.append(i)
            elif 380 <= len(i["output"]) < 800:
                M.append(i)
            elif 800 <= len(i["output"]):
                L.append(i)
        final_S = S
        final_M = M
        final_L = L
        category_1 = final_S + final_M + final_L
        if hash_container:
            return set(category_1)
        return category_1


class CategoryTwo:
    def __init__(self, data):
        self.data = data
        self.container = list()

    def search_data_2(self):
        c = 0  # theory|explain|reason|physics|formula|calculate|why
        for n, i in enumerate(self.data, start=1):
            x = re.search(
                r"\b(?:physics|chemistry|biology|logic|philosophy|explain|why|law)\b",
                i["instruction"], re.IGNORECASE)
            if x and "http" not in i["input"] and "http" not in i["output"] and "http" not in i[
                "instruction"] and "invoke" not in i['instruction'].lower():
                if len(i["output"]) >= 200 and not ("arrested for" in i['instruction'].lower() or "capital of" in i[
                    'instruction'].lower() or "headline for" in i['instruction'].lower() or "summarize the following" in
                                                    i['instruction'].lower() or "zip code" in i[
                                                        'instruction'].lower() or "population" in i[
                                                        'instruction'].lower()):
                    logic_hard = ["because", "therefore", "consequently", "thus", "due to", "as a result",
                                  "for instance", "moreover", "specifically"]
                    logic_soft = ["if", "so", "then", "however", "but", "instead", "while", "actually"]
                    has_hard = any(m in i["output"].lower() for m in logic_hard)
                    soft_count = sum(1 for m in logic_soft if m in i["output"].lower())
                    if has_hard or soft_count >= 2:
                        if len(i["output"]) >= 200:
                            self.container.append(i)
                            c += 1

    def final_filter(self, hash_container=False):
        S, M, L = list(), list(), list()
        for i in self.container:
            if 200 <= len(i["output"]) < 460:
                S.append(i)
            elif 460 <= len(i["output"]) < 900:
                M.append(i)
            elif 900 <= len(i["output"]):
                L.append(i)
        final_S = S  # random.sample(S, 560)
        final_M = M  # random.sample(M, 570)
        final_L = L
        category_2 = final_S + final_M + final_L
        if hash_container:
            return set(category_2)
        return category_2


class CategoryThree:
    def __init__(self, data):
        self.data = data
        self.container = list()

    def search_data(self):
        c = 0  # theory|explain|reason|physics|formula|calculate|why
        for n, i in enumerate(self.data, start=1):
            x = re.search(
                r"\b(?:strategy|marketing|business plan|proposal|investment|swot|startup|roi|revenue|market analysis|roadmap|market research|target audience|B2B|SaaS|milestones)\b",
                i["instruction"], re.IGNORECASE)
            if x and "http" not in i["input"] and "http" not in i["output"] and "http" not in i["instruction"]:
                self.container.append(i)
                c += 1

    def final_filter(self, hash_container=False):
        container_biz = []
        business_markers = ["strategy", "marketing", "business", "proposal", "investment", "startup", "revenue",
                            "market", "customer", "product"]
        for i in self.container:
            instr = i["instruction"].lower()
            out = i["output"].lower()
            if any(m in instr for m in business_markers):
                if "http" not in i["output"] and "refactor" not in instr:
                    if len(i["output"]) >= 200:
                        logic_markers = ["because", "therefore", "however", "so", "then", "instead", "if"]
                        has_logic = any(lm in out for lm in logic_markers)

                        if has_logic or len(i["output"]) > 570:
                            container_biz.append(i)
        category_3 = container_biz
        if hash_container:
            return set(category_3)
        return category_3


class CategoryFour:
    def __init__(self, data):
        self.data = data
        self.container = list()

    def search_data(self):
        c = 0
        for n, i in enumerate(self.data, start=1):
            x = re.search(
                r"\b(?:design|write|creative|brand|art|logo|story|poem|novel|lyrics|slogan|metaphor|inspiration|plot|fiction|style of|rewrite as|tone|persona|impersonate|storytelling|dialogue|narrative|creative description|polished|sophisticated|elegant|eloquent)\b",
                i["instruction"], re.IGNORECASE)
            if x and "http" not in i["input"] and "http" not in i["output"] and "http" not in i["instruction"]:
                self.container.append(i)
                c += 1

    def final_filter(self, hash_container=False):
        S, M, L = list(), list(), list()
        new_container = (i for i in self.container if 280 <= len(i["output"]) <= 1200)
        for i in new_container:
            if 280 <= len(i["output"]) < 500:
                S.append(i)
            elif 500 <= len(i["output"]) < 650:
                M.append(i)
            elif 650 <= len(i["output"]) < 1200:
                L.append(i)
        final_S = random.sample(S, 400)
        final_M = random.sample(M, 600)
        final_L = random.sample(L, 200)
        category_4 = final_S + final_M + final_L
        if hash_container:
            return set(category_4)
        return category_4


class CategoryFive:
    def __init__(self, data):
        self.data = data
        self.container = list()

    def search_data(self):
        c = 0
        for n, i in enumerate(self.data, start=1):
            x = re.search(
                r"\b(?:travel|cooking|recipe|fitness|health|diet|cleaning|home|workout|grocery|nutrition|sleep|routine|schedule|reminder|appointment|medication|biohacking|productivity|flight|hotel|reservation|exercise)\b",
                i["instruction"], re.IGNORECASE)
            if x and "http" not in i["input"] and "http" not in i["output"] and "http" not in i["instruction"]:
                self.container.append(i)
                c += 1

    def final_filter(self, hash_container=False):
        S, M, L = list(), list(), list()
        new_container = list(i for i in self.container if 200 <= len(i["output"]) <= 1000)
        for i in new_container:
            if 200 <= len(i["output"]) < 450:
                S.append(i)
            elif 450 <= len(i["output"]) < 750:
                M.append(i)
            elif 751 <= len(i["output"]) <= 1000:
                L.append(i)
        final_S = random.sample(S, 300)
        final_M = M
        final_L = L
        category_5 = final_S + final_M + final_L
        if hash_container:
            return set(category_5)
        return category_5


class CategorySix:
    def __init__(self, data):
        self.data = data
        self.container = list()

    def search_data(self):
        c = 0
        for n, i in enumerate(self.data, start=1):
            x = re.search(
                r"\b(?:ethical|harmful|dangerous|safety|security|hack|exploit|illegal|protect|hazard|risk|warning|violation|privacy|prohibited|unauthorized)\b",
                i["instruction"], re.IGNORECASE)
            if x and "http" not in i["input"] and "http" not in i["output"] and "http" not in i["instruction"]:
                self.container.append(i)
                c += 1

    def final_filter(self, hash_container=False):
        category_6 = list(i for i in self.container if 150 <= len(i["output"]) <= 1200)
        if hash_container:
            return set(category_6)
        return category_6


class CategorySeven:
    def __init__(self, data):
        self.data = data
        self.container = list()

    def search_data(self):
        c = 0
        for n, i in enumerate(self.data, start=1):
            x = re.search(
                r"\b(?:irony|sarcasm|witty|banter|sassy|mischief|wit|humor|joke|insult|loyal|attitude)\b"
                , i["instruction"], re.IGNORECASE)
            if x and "http" not in i["input"] and "http" not in i["output"] and "http" not in i["instruction"]:
                self.container.append(i)
                c += 1

    def sort(self, reverse=False):
        self.container = sorted(self.container, key=lambda x: len(x["output"]), reverse=reverse)

    def final_filter(self, hash_container=False):
        category_7 = list(i for i in self.container if 100 <= len(i["output"]) <= 920)
        if hash_container:
            return set(category_7)
        return category_7


class FindDublicate:
    def __init__(self, data):
        self.c = 0
        self.filtred_cat_colector = []
        self.alpaca_data = data
        self.collector = set()

    def extract(self, obj_class):
        obj = obj_class(self.alpaca_data)
        obj.search_data()
        return obj.final_filter()

    def start(self):
        for cat in (CategoryOne, CategoryTwo, CategoryThree, CategoryFour, CategoryFive, CategorySix, CategorySeven):
            filterd_cat = []
            for i in self.extract(cat):
                if i["instruction"] in self.collector:
                    self.c += 1
                else:
                    self.collector.add(i["instruction"])
                    filterd_cat.append(i)
            self.filtred_cat_colector.append(filterd_cat)
        return self.filtred_cat_colector

def ready_to_generate():
    data = load_dataset("tatsu-lab/alpaca")
    data = data["train"]  # .shuffle(seed=560).select(range(10))  
    
    find_duplicate = FindDuplicate(data)
    ready_cats = find_duplicate.start()  # явно передаем
    return ready_cats

def main():
    filtred_cats = ready_to_generate(data)
    for cat in filtred_cats:
        print(len(cat))

if __name__ == "__main__":
    main()

