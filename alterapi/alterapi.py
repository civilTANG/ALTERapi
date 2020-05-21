import ast, astor, re
import os, subprocess
import builtins
builtin_types = [getattr(builtins, d) for d in dir(builtins) if isinstance(getattr(builtins, d), type)]

# code template to be instrumented
template = """
import sys, timeit, string
from itertools import chain
# original code
r1 = {}  
# necessary initializations
{}
# replaced code
r2 = {}
# assert output equivalence
if isinstance(r1, pd.Series) and isinstance(r2, pd.Series) or isinstance(r1, pd.DataFrame) and isinstance(r2, pd.DataFrame):
    assert r1.equals(r2)
elif isinstance(r1, pd.DataFrame) and isinstance(r2, np.ndarray) or isinstance(r2, pd.DataFrame) and isinstance(r1, np.ndarray):
    df_data = r1 if isinstance(r1, pd.DataFrame) else r2
    nd_data = pd.DataFrame(r1) if isinstance(r1, np.ndarray) else pd.DataFrame(r2)
    assert df_data.equals(nd_data)
else:
    assert np.array_equal(r1, r2) == True


def func1():
    {}


def func2():
    {}

# get execution time
number_of_runs = 100
t1 = timeit.timeit(func1, number=number_of_runs) / number_of_runs
t2 = timeit.timeit(func2, number=number_of_runs) / number_of_runs
with open('optimization.csv', 'w') as f:
    writer = f.write(str(t1) + ',' + str(t2))
sys.exit(0)
"""


class CallParser(ast.NodeVisitor):
    def __init__(self):
        self.attrs = []
        self.names = []

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
        if isinstance(node.func, ast.Attribute) and node.func.attr in target_apis:
                name = node.func.attr
                self.names.append(name)
                self.attrs.append(node)
        elif isinstance(node.func, ast.Name) and node.func.id in target_apis:
                name = node.func.id
                self.names.append(name)
                self.attrs.append(node)

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Attribute) and node.value.attr in target_apis:
            name = node.value.attr
            self.names.append(name)
            self.attrs.append(node)



class CodeInstrument(ast.NodeTransformer):
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
                        value.insert(index + 1, self.newnode)
                        return node
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)


target_apis = ['iloc', 'loc', 'where', 'apply', 'map', 'replace',
               'sum', 'nonzero', 'einsum', 'dot', 'full', 'empty',
               'vstack', 'c_', 'hstack', 'column_stack', 'array', 'vectorize',
               'atleast_2d', 'combine_first', 'crosstab', 'to_datetime',
               'unstack', 'groupby', 'value_counts', 'cumprod']

direct_replaces = {'replace': 'map',
                   'vstack': 'concatenate',
                   'combine_first': 'fillna',
                   'column_stack': 'transpose'}

compare_direct_replaces = {"sum": "count_nonzero",
                           "nonzero": "where"}

class APIReplace(object):
    def __init__(self, code_path, option='static'):
        self.code_path = code_path
        self.option = option

    def recommend(self):
        cnt = 0

        with open(self.code_path, 'r', encoding='UTF-8') as file:
            lines = file.readlines()
        with open(self.code_path, 'r', encoding='UTF-8') as file:
            content = file.read()

        tree = ast.parse(content)
        v = CallParser()   # find the potential api
        v.visit(tree)
        names = v.names

        for index, candidate in enumerate(v.attrs):
            oldstmt = astor.to_source(candidate).strip()
            target_api = names[index]
            lineno = candidate.lineno
            newstmt, env = None, ""

            if isinstance(candidate, ast.Subscript) and target_api in ['loc', 'iloc', 'c_']:
                if isinstance(candidate.slice.value, ast.Tuple):
                    if target_api == "c_":
                        replace_api = "hstack"
                    else:
                        replace_api = "at" if isinstance(candidate.slice.value.elts[1], ast.Str) else "iat"
                else:
                    replace_api = "iloc"

                newstmt = oldstmt.replace(target_api, replace_api)
                if target_api == "c_":
                    newstmt = newstmt.replace("[","((").replace("]","))")
                self.__run(oldstmt, env, newstmt, content, lineno, cnt)
                continue

            receiver = astor.to_source(candidate.func.value).strip()
            args_cnt = len(candidate.args) + len(candidate.keywords)

            if len(candidate.args) == 0:
                if target_api == "sum":
                    if len(candidate.keywords) == 0 and isinstance(candidate.func.value, ast.Compare):
                        newstmt = "np.count_nonzero{}".format(receiver)
                    elif "*" in receiver and candidate.keywords[0].arg == "axis":
                        var1, var2 = receiver[1:-1].split("*")
                        newstmt = "np.einsum('...i,...i ->...', {}, {})".format(var1, var2)
                if target_api in ["unstack", "filter"]:
                    vs = CallParser()
                    vs.visit(ast.parse(oldstmt))
                    if vs.names == ['groupby', 'value_counts', 'unstack']:
                        col1 = candidate.func.value.func.value.value.args[0].s
                        col2 = candidate.func.value.func.value.slice.value.s
                        newstmt = "df.groupby(['{}','{}']).size().unstack(fill_value=0)".format(col1, col2)
                    if vs.names == ['groupby', 'filter']:
                        newstmt = ""
                if target_api == "cumprod":
                    axis = 0
                    if len(candidate.keywords) == 1 and candidate.keywords[0].arg == "axis":
                        axis = candidate.keywords[0].value.n
                    newstmt = "np.cumprod(axis={})".format(axis)
                if newstmt:
                    self.__run(oldstmt, env, newstmt, content, lineno, cnt)
                continue

            arg = candidate.args[0]
            argstr = arg.s if isinstance(arg, ast.Str) else astor.to_source(arg).strip()

            if target_api in direct_replaces and args_cnt == 1:
                newstmt = oldstmt.replace(target_api, direct_replaces[target_api])

            elif target_api == "map" and isinstance(arg, ast.Name) and type(arg.id) in builtin_types:
                newstmt = oldstmt.replace(target_api, "astype")

            elif isinstance(arg, ast.Compare):
                if target_api in compare_direct_replaces:
                    newstmt = oldstmt.replace(target_api, compare_direct_replaces[target_api])
                elif target_api == "where":
                    if receiver == "np" and args_cnt == 3 and candidate.args[1].n in [0, 1] and candidate.args[2].n in [0, 1]:
                        newstmt = "({}).astype(int)".format(argstr)
                    elif lines[lineno-1].strip().endswith("[0][0]"):
                        newstmt = "np.argmax({})".format(argstr)
                        oldstmt = lines[lineno-1].strip()
                    else:
                        cond = astor.to_source(candidate.args[0]).strip().replace(receiver, receiver + ".values")
                        other = astor.to_source(candidate.args[1]).strip() if len(candidate.args) == 2 else "np.nan"
                        newstmt = "np.where({}, {}.values, {})".format(cond, receiver, other)

            elif isinstance(arg, ast.Lambda):
                if not hasattr(arg.body, "test") and target_api == "apply":
                    if isinstance(arg.body, ast.Call) and type(arg.body.func.id) in builtin_types:
                        newstmt = "{}.astype({})".format(receiver, arg.body.func.id)
                    else:
                        newstmt = oldstmt.replace(target_api, "map")
                else:
                    test_clause = astor.to_source(arg.body.test).strip()
                    var_clause= astor.to_source(arg.args).strip()
                    test_clause = test_clause[len(var_clause):].strip()
                    if_clause = astor.to_source(arg.body.body).strip()
                    else_clause = astor.to_source(arg.body.orelse).strip()
                    if args_cnt == 1 and target_api == "map":
                        newstmt = "np.where({}{}, {},{})".format(receiver, test_clause, if_clause, else_clause)
                    elif target_api == "apply" and args_cnt == 2:
                        newstmt = "np.where({}{}, {},{})".format(receiver + test_clause[test_clause.index("["):test_clause.index("]")+1], test_clause[test_clause.index("]")+1:], if_clause, else_clause)

            elif target_api == "sum":
                check = "string.ascii_lowercase[:len({}.shape)]".format(argstr)
                newstmt = "np.einsum({} + '->', {})".format(check, argstr)

            elif target_api == "einsum":
                subscripts = re.split(",|->", argstr)
                params = [x.id for x in candidate.args[1:]]
                if len(subscripts) == 3 and subscripts[0][-1] == subscripts[1][-2]:
                    if subscripts[2] == subscripts[1][-1]:
                        newstmt = "np.einsum('j,jk->k', np.einsum('ij->j', {}), {})".format(params[0], params[1])
                    else:
                        newstmt = "np.dot({})".format(",".join(params))

            elif target_api == "dot":
                part1 = "part1 = string.ascii_lowercase[:len({}.shape)]\n".format(receiver)
                part2 = "part2 = string.ascii_lowercase[::-1][:len({}.shape)]\n".format(argstr)
                part2 += "part2 = part2[:-2] + part1[-1] + part2[-1]\n"
                part3 = "part3 = part1[:-1] + part2[:-2] + part2[-1]\n"
                env = part1 + part2 + part3
                newstmt = "np.einsum(part1+',' + part2 + '->' + part3, {},{})".format(receiver, argstr)

            elif target_api == "hstack":
                if isinstance(arg, ast.Name):
                    newstmt = "np.array(list(chain.from_iterable({})))".format(argstr)
                else:
                    newstmt = "np.concatenate({}, axis=1)".format(argstr)

            elif target_api == "atleast_2d":
                newstmt = "{}.reshape(1,-1)".format(argstr)

            elif target_api == "array":
                newstmt = "np.fromiter({},dtype={})".format(argstr, "type({}[0])".format(argstr))
                if isinstance(arg, ast.Call) and arg.func.id == "range":
                    params = [str(x.n) for x in arg.args]
                    newstmt = "np.arange({})".format(",".join(params))

            elif target_api == "crosstab" and args_cnt == 2:
                newstmt = "{}.pivot_table(index='{}', columns='{}', aggfunc=len, fill_value=0)".format(arg.value.id, arg.slice.value.s, candidate.args[1].slice.value.s)

            elif target_api == "empty":
                thisline = lines[lineno-1].replace(" ", "").strip()
                nextline = lines[lineno].replace(" ", "").strip() if len(lines) > lineno else None
                if thisline.endswith("[:]=0") or thisline.endswith("[:]=1"):
                    oldstmt = thisline
                elif nextline and (nextline.endswith("[:]=0") or nextline.endswith("[:]=1")):
                    oldstmt += ";" + nextline
                newstmt = "np.zeros({})".format(argstr) if oldstmt.endswith("0") else "np.ones({})".format(argstr)

            elif target_api == "full":
                if candidate.args[1].n == 0:
                    newstmt = "np.zeros({})".format(argstr)
                elif candidate.args[1].n == 1:
                    newstmt = "np.ones({})".format(argstr)

            elif target_api == "vectorize":
                tmptree = ast.parse(lines[lineno-1].strip())
                receiver = tmptree.body[0].value.args[0].id
                oldstmt += "({})".format(receiver)
                newstmt = "np.frompyfunc({}, 1, 1)({})".format(argstr, receiver)

            if newstmt:
                self.__run(oldstmt, env, newstmt, content, lineno, cnt)
                cnt += 1

    def __run(self, oldstmt, env, newstmt, content, lineno, cnt):
        if self.option == 'static':
            print("Code at line {}  : {}".format(lineno, oldstmt))
            print('Recommended code: {}\n'.format(newstmt))
        elif self.option == 'dynamic':
            to_add = template.format(oldstmt, env, newstmt, oldstmt, newstmt, lineno, '"{}"'.format(oldstmt),
                                     '"{}"'.format(newstmt))

            # fill in the template
            to_insert = ast.parse(to_add)
            # insert the new node
            temp_tree = ast.parse(content)
            CodeInstrument(lineno, to_insert).visit(temp_tree)
            new_source = astor.to_source(temp_tree)

            wd = os.getcwd()
            cache_path = os.path.join(wd, 'alterapi_cache')
            # create a new file
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            des_path = os.path.join(cache_path, 'code_{}.py'.format(cnt))
            with open(des_path, 'w') as wf:
                wf.write(new_source)

            cmd = "python " + des_path
            try:

                os.chdir(cache_path)
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, err = p.communicate(timeout=600)
                os.chdir(wd)

                if p.returncode == 1:
                    #print("execution interrupt:" + err.decode(sys.stdout.encoding))
                    return

                with open(os.path.join(cache_path, "optimization.csv"), "r") as f:
                    text = f.read().strip()
                    t1, t2 = float(text.split(",")[0]), float(text.split(",")[1])
                    if t2 < t1:
                        print("Code at line {} : {}".format(lineno, oldstmt))
                        print('Recommended code: {}'.format(newstmt))
                        print('original time:{:.1e}s, new time:{:.1e}s, speedup:{:.1f}x'.format(t1, t2,
                                                                                                float(t1) / float(t2)))
                    print("----------------------------------------------------------------------------")

            except Exception as e:
                print(e)
                return


if __name__ == "__main__":
    tool = APIReplace('../tests/input.py', option='dynamic')
    tool.recommend()












