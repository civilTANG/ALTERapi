import sys, os
import ast, astor

# root directory
root_dir = "C://Users//Yida//kaggle//kaggle_scripts//crawled_data//"

# Candidate for replacement
api_full = "np.arange"
api_simple = "arange"
api_alternative = "np.array"

# template for replacement
template = """    
import sys, timeit, csv\n

r1 = {} # original code \n
r2 = {}  # api replace \n

assert np.allclose(r1,r2) \n

def func1():\n
    {}\n
def func2():\n
    {}\n

number_of_runs = 100 \n
t1 = timeit.timeit(func1, number=number_of_runs) / number_of_runs\n
t2 = timeit.timeit(func2, number=number_of_runs) / number_of_runs\n

with open("../link.txt") as f:\n
    url = f.readlines()[0].strip()\n
with open("../../../optimization.csv", "a+") as f:\n
    writer = csv.writer(f)\n
    writer.writerow([url, {}, {}, {}, t1, t2, {}, {}])\n
sys.exit(0)\n
"""

# get node that calls the target API
from ast import Attribute, Name
class CallParser(ast.NodeVisitor):
    def __init__(self):
        self.attrs = []
        self.name = ""

    def generic_visit(self, node):
        rlist = list(ast.iter_fields(node))
        for field, value in reversed(rlist):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)


    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, Attribute) and node.func.attr == api_simple:
            self.attrs.append(node)
        elif isinstance(node.func, Name) and node.func.id == api_simple:
            self.attrs.append(node)

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, Attribute) and node.value.attr == api_simple:
            self.attrs.append(node)


# insert new code
class CodeInstrumentator(ast.NodeTransformer):
    def __init__(self, lineno, newnode):
        self.line = lineno
        self.newnode = newnode

    def generic_visit(self, node):
        rlist = list(ast.iter_fields(node))
        for field, value in reversed(rlist):
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, 'lineno') and item.lineno == self.line:
                        index = value.index(item)
                        value.insert(index+1, self.newnode)
                        return node
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)



parsed_links = []
parsed_scripts = []
total = 0
api_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
# search folders
for apifolder in api_folders:
    src_folders = [d for d in os.listdir(apifolder)]

    for srcfolder in src_folders:

        # skip same scripts with same urls
        with open(os.path.join(root_dir, apifolder, srcfolder, "link.txt")) as f:
            url = f.readlines()[0].strip()
            if url in parsed_links:
                continue
            parsed_links.append(url)

        code_path = os.path.join(root_dir, apifolder, srcfolder, "code", 'code.py')
        try:
            with open(code_path, 'r') as file:
                content = file.read()

            # skip duplicated scripts with different urls
            chash = abs(hash(content)) % (10 ** 8)
            if chash in parsed_scripts:
                continue
            parsed_scripts.append(chash)

            if api_simple not in content:
                continue

            tree = ast.parse(content)
            v = CallParser()
            v.visit(tree)

            cnt = 0
            for candidate in v.attrs:
                oldstmt = astor.to_source(candidate).strip()
                lineno = candidate.lineno
                # API replacement
                newstmt = oldstmt.replace("arange", "array(range") + ")"

                # fill in the template
                to_add = template.format(oldstmt, newstmt, oldstmt, newstmt, lineno, '"{}"'.format(oldstmt), '"{}"'.format(newstmt), "'{}'".format(api_full), "'{}'".format(api_alternative))
                to_insert = ast.parse(to_add)
                # insert the new node
                CodeInstrumentator(lineno, to_insert).visit(tree)
                instru_source = astor.to_source(tree)

                des_path = os.path.join(root_dir, apifolder, srcfolder, "code", '{}_replace{}.py'.format(api_full, cnt))
                with open(des_path, 'w') as wf:
                    print(des_path)
                    wf.write(instru_source)
                cnt += 1
                total += 1
        except:
            continue

print(total)

