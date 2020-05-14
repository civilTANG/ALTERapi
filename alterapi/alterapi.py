import ast,astor, astunparse
from ast import Attribute, Name
import os, csv
import subprocess
api_name = ['c_', 'iloc',  'loc', 'apply','sum', 'count_nonzero','hstack','array', 'where', 'transpose', 'query',
            'vstack','zeros','query','apply','map','crosstab','where','atleast_2d','tile','loc',
            'to_datetime','array','hstack','concatenate','iloc','astype','array','str',
            'fillna','norm','where','array','column_stack','count_nonzero','nonzero','transpose',
            'replace','cumprod','tensordot','iterrows','query','arange','iat','argmax',
            'array','column_stack','dot','full','hstack','ones','vstack','zeros','ix','dot','at','append']

# direct replacement
api_pair0 = {'astype': ['apply','map'], 'column_stack': ['transpose'],
            'loc': ['ix'], 'transpose': ['column_stack'],
            'ix': ['loc'], 'nonzero': ['where'], 'apply': ['map'],
             'query': ['loc'], 'replace': ['map'],
            'at': ['loc'], 'iloc': ['loc'], 'iat': ['iloc'],
            'map': ['apply'], 'fillna': ['combine_first'],
             'hstack': ['c_'], 'array': ['hstack'], 'fromiter': ['array'] }

api_pair1 ={'where': ['nonzero'],  'hstack': ['append', 'concatenate'],
           'vstack': ['concatenate', 'column_stack'],
            'column_stack': ['vstack'], 'array': ['fromiter', 'concatenate'], 'zeros': ['empty'], 'ones': ['empty'],
            'map': ['replace', 'where'], 'str': ['map'], 'c_': ['hstack'], 'arange':['array'],'count_nonzero': ['sum'],
            'sum': ['count_nonzero', 'sum', 'einsum']}

api_pair2 = {'full': ['empty', 'zeros'], 'dot': ['einsum','tensordot'],'sum': ['sum']}

api_pair3 = {'where': ['apply', 'map', 'astype']}


# template for replacement
template = """    
import os, timeit, csv\n
r1 = {} # original code \n
r2 = {}  # api replace \n 

     
if (isinstance(r1, pd.Series) and isinstance(r2, pd.Series)) or (isinstance(r1, pd.DataFrame) and isinstance(r2, pd.DataFrame)):\n
    assert r1.equals(r2)\n   
else:\n
    assert np.allclose(r1,r2)\n
def func1():\n
    {}\n
def func2():\n
    {}\n
number_of_runs = 100 \n
t1 = timeit.timeit(func1, number=number_of_runs) / number_of_runs\n
t2 = timeit.timeit(func2, number=number_of_runs) / number_of_runs\n
with open("optimization.csv", "a+") as f:\n
     writer = csv.writer(f)\n
     writer.writerow([{},{},{},t1, t2])\n
os._exit(0)\n
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
        if isinstance(node.func, Attribute) and node.func.attr in api_name:
                name = node.func.attr
                self.names.append(name)
                self.attrs.append(node)
        elif isinstance(node.func, Name) and node.func.id in api_name:
                name = node.func.id
                self.names.append(name)
                self.attrs.append(node)

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, Attribute) and node.value.attr in api_name:
            name = node.value.attr
            self.names.append(name)
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
                        value.insert(index + 1, self.newnode)
                        return node
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)


class APIReplace(object):
    def __init__(self, code_path, option='static'):
        self.code_path = code_path
        self.option = option

    def recommend(self):
        cnt = 0
        index = 0
        # open file
        with open(self.code_path, 'r', encoding='UTF-8') as file:
            content = file.read()
        file.close()

        # parse
        tree = ast.parse(content)  # parse the code into ast_tree
        v = CallParser()   # find the potential api
        v.visit(tree)
        names = v.names

        for candidate in v.attrs:
                oldstmt = astor.to_source(candidate).strip()
                lineno = candidate.lineno

                if names[index] in api_pair0.keys():
                    for i in range(0, len(api_pair0[names[index]])):
                        newstmt = oldstmt.replace(names[index], api_pair0[names[index]][i])
                        print("original API:" + oldstmt)
                        print("lineno:{}".format(lineno))
                        if self.option == 'dynamic':
                            self.add_source(oldstmt, newstmt, content, lineno, cnt)
                            cnt += 1
                        print('Recommend API:' + newstmt)
                        print("----------------------------------------------------------------------------")

                if isinstance(candidate, ast.Call):
                    number_agrs = len(candidate.args)
                    keywords = candidate.keywords
                    objt = candidate.func.value
                    if number_agrs == 1:
                        agr1 = candidate.args
                        if names[index] in api_pair1.keys():
                            for i in range(0, len(api_pair1[names[index]])):
                                newstmt = self.replace_1agr(oldstmt, agr1, keywords, objt, names[index], api_pair1[names[index]][i])
                                if newstmt:
                                    print("original API:" + oldstmt)
                                    print("lineno:{}".format(lineno))
                                    if self.option == 'dynamic':
                                        self.add_source(oldstmt, newstmt, content, lineno, cnt)
                                        cnt += 1
                                    print('Recommend API:' + newstmt)
                                    print("----------------------------------------------------------------------------")

                    elif number_agrs == 2:
                        agr1, agr2 = candidate.args
                        if names[index] in api_pair2.keys():
                            for i in range(0, len(api_pair2[names[index]])):
                                newstmt = self.replace_2agr(oldstmt, agr1, agr2, keywords, names[index], api_pair2[names[index]][i])
                                if newstmt:
                                    print("original API:" + oldstmt)
                                    print("lineno:{}".format(lineno))
                                    if self.option == 'dynamic':
                                        self.add_source(oldstmt, newstmt, content, lineno, cnt)
                                        cnt += 1
                                    print('Recommend API:' + newstmt)
                                    print("----------------------------------------------------------------------------")
                    elif number_agrs == 3:
                        agr1, agr2, agr3 = candidate.args
                        if names[index] in api_pair3.keys():
                            for i in range(0, len(api_pair3[names[index]])):
                                newstmt = self.replace_3agr(oldstmt, agr1, agr2, agr3, names[index], api_pair3[names[index]][i])
                                if newstmt:
                                    print("original API:" + oldstmt)
                                    print("lineno:{}".format(lineno))
                                    if self.option == 'dynamic':
                                        self.add_source(oldstmt, newstmt, content, lineno, cnt)
                                        cnt += 1
                                    print('Recommend API:' + newstmt)
                                    print("----------------------------------------------------------------------------")

                    else:
                        print('No ability deal with this situation')
                        print("----------------------------------------------------------------------------")

                elif isinstance(candidate, ast.Subscript):
                    if isinstance(candidate.slice, ast.Index):
                        keywords = []
                        if names[index] in api_pair1.keys():
                            for i in range(0, len(api_pair1[names[index]])):
                                newstmt = self.replace_1agr(oldstmt, candidate.slice, keywords, names[index], api_pair1[names[index]][i])
                                if newstmt:
                                    print("original API:" + oldstmt)
                                    print("lineno:{}".format(lineno))
                                    print('Recommend API:' + newstmt)
                                    print("----------------------------------------------------------------------------")
                                    if self.option == 'dynamic':
                                        self.add_source(oldstmt, newstmt, content, lineno, cnt)
                                        cnt += 1

                    else:
                        print('No ability deal with this situation')
                        print("----------------------------------------------------------------------------")
                index = index + 1


    def add_source(self, oldstmt, newstmt, content, lineno, cnt):
        to_add = template.format(oldstmt, newstmt, oldstmt, newstmt, lineno, '"{}"'.format(oldstmt),
                                 '"{}"'.format(newstmt))
        # fill in the template
        to_insert = ast.parse(to_add)
        # insert the new node
        temp_tree = ast.parse(content)
        CodeInstrumentator(lineno, to_insert).visit(temp_tree)
        instru_source = astor.to_source(temp_tree)
        des_path = os.path.join(os.path.dirname(self.code_path), 'code_{}.py'.format(cnt))
        with open(des_path, 'w') as wf:
            wf.write(instru_source)
        wf.close()
        self.execute_code(des_path)

    def execute_code(self, des_path):
        try:
            cmd = "python " + des_path
            print("executing:" + des_path)
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, err = p.communicate(timeout=600)
            err = str(err).split('\\r\\n')
            if p.returncode == 1:
                error_type = err[len(err) - 2]
                print("execution interrupt:" + error_type)
            else:
                with open("optimization.csv", "r") as f:
                    reader = csv.reader(f)
                    rows = [row for row in reader]
                    row = rows[-2]
                    t1 = row[3]
                    t2 = row[4]
                    print('origin time:{} , recommend time:{}'.format(t1, t2))
                f.close()
        except Exception as e:
                print(e)

    def replace_1agr(self,oldstmt , agr1, keywords, objt, name, target_name):
        if isinstance(agr1, ast.Index):
            inx = astunparse.unparse(agr1)
        else:
            agr_one = astor.to_source(agr1[0]).strip()

        if name == 'hstack' and (isinstance(agr1[0], ast.Tuple) or isinstance(agr1[0], ast.List)):
            if len(agr1[0].elts) == 2 and target_name == 'append' and isinstance(agr1[0], ast.Tuple):
                newstmt = 'np.' + target_name + agr_one
                return newstmt
            if target_name == 'concatenate':
                newstmt = 'np.{}({}, axis=1)'.format(target_name, agr_one)
                return newstmt

        elif name == 'where' or (name == 'map' and isinstance(agr1[0], ast.Dict))\
                or (name == 'count_nonzero' and isinstance(agr1[0], ast.Compare))\
                or (name == 'sum' and isinstance(agr1[0], ast.Compare) and target_name == 'count_nonzero'):
            newstmt = oldstmt.replace(name, target_name)
            return newstmt

        elif name == 'vstack' and (isinstance(agr1[0], ast.Tuple) or isinstance(agr1[0], ast.List)):
            if target_name == 'concatenate':
                newstmt = 'np.{}({},axis=0)'.format(target_name, agr_one)
                return newstmt
            elif target_name == 'column_stack':
                newstmt = 'np.{}({}).T'.format(target_name, agr_one)
                return newstmt
        elif name == 'column_stack':
            newstmt = 'np.{}({}).T'.format(target_name, agr_one)
            return newstmt

        elif name == 'array':
            if len(keywords) == 0 and target_name == 'concatenate':
                newstmt = 'np.{}([{}])'.format(target_name, agr_one)
                return newstmt
            elif len(keywords) != 0:
                if keywords[0].arg == 'dtype' and target_name == 'fromiter':
                    newstmt = oldstmt.replace(name, target_name)
                    return newstmt
            else:
                pass
        elif name == 'zeros':
            newstmt = oldstmt.replace(name, target_name) + "; r2[:]= 0"
            return newstmt

        elif name == 'ones':
            newstmt = oldstmt.replace(name, target_name) + "; r2.fill(1)"
            return newstmt

        elif name == 'sum':
            if target_name == 'sum':
                newstmt = '({}).{}()'.format(agr_one, target_name)
                return newstmt
            if target_name == 'einsum':
                if len(keywords) == 0:
                    newstmt = "np.{}('i->',{})".format(target_name, agr_one)
                    return newstmt
                else:
                    keyword = astunparse.unparse(keywords[0])
                    if ('axis=1' in keyword) or ('axis=-1' in keyword):
                        newstmt = "np.{}('ij->i',{} )".format(target_name, agr_one)
                        return newstmt
                    elif ('axis=-2' in keyword) or ('axis=0' in keyword):
                        newstmt = "np.{}('ij->j',{} )".format(target_name,agr_one)
                        return newstmt

        elif name == 'str':
            newstmt = oldstmt.replace('str', 'map(lambda x: x') + ')'
            return newstmt

        elif name == 'c_':
            newstmt = oldstmt.replace("c_", "hstack(") + ")"
            return newstmt

        elif name == 'arrange':
            newstmt = oldstmt.replace('arrange(', 'np.array(range(')
            return newstmt

            # pd.Series.map	-> np.where
        elif name == 'map' and target_name == 'where':
            if isinstance(agr1[0], ast.Lambda):
                if isinstance(agr1[0].body, ast.IfExp):
                    cond = astunparse.unparse(agr1[0].body.test)
                    cond = cond.replace(astunparse.unparse(agr1[0].args.args).strip(),
                                        astunparse.unparse(objt).strip()).strip()
                    newstmt = 'np.{}({},{},{})'.format(target_name, cond, astunparse.unparse(agr1[0].body.body).strip(),
                                                      astunparse.unparse(agr1[0].body.orelse).strip())
                    return newstmt

    def replace_2agr(self, oldstmt, agr1, agr2, keywords, name, target_name):
        agr_one = astor.to_source(agr1).strip()
        arg_two = astor.to_source(agr2).strip()
        if name == 'full':
            if len(keywords) != 0:
                if isinstance(keywords[0], ast.keyword) and keywords[0].arg == 'dtype':
                    keyword = astunparse.unparse(keywords[0])
                    if not (target_name == 'zeros' and int(arg_two) != 0):
                        newstmt = 'np.{}({},{});r2[:] ={}'.format(target_name,agr_one,keyword,arg_two).replace('\n','')
                        return newstmt
            else:
                if not (target_name == 'zeros' and int(arg_two) != 0):
                        newstmt = 'np.{}({});r2[:] ={}'.format(target_name, agr_one, arg_two).replace('\n','')
                        return newstmt

        elif name == 'dot':
            if target_name == 'einsum':
                newstmt = "np.{}('ij,jm->im',{},{} )".format(target_name,agr_one,arg_two)
                return newstmt
            elif target_name == 'tensordot':
                newstmt = 'np.{}({},{},axes=1)'.format(target_name,agr_one,arg_two)
                return newstmt

        elif name == 'sum' and target_name == 'sum':
            newstmt = '({}).{}({})'.format(agr_one, target_name, arg_two)
            return newstmt

    def replace_3agr(self, oldstmt, agr1, agr2, agr3, name, target_name):
        if name == 'where':
            if isinstance(agr1, ast.Compare):
                obj = astor.to_source(agr1.left).strip()
                idname = agr1.left.value.id
                attr = agr1.left.attr
                agr_one = astor.to_source(agr1).strip()
                arg_two = astor.to_source(agr2).strip()
                arg_three = astor.to_source(agr3).strip()
                if target_name == 'map':
                    cond = astor.to_source(agr1).strip().replace(obj, "x")
                    newstmt = "{}.{}(lambda x: {} if {} else {})".format(obj, target_name, arg_two, cond, arg_three)
                    return newstmt
                elif target_name == 'apply':
                    cond = astor.to_source(agr1).strip().replace(obj, '#')
                    attr = "row['" + attr + "']"
                    cond = cond.replace('#', attr)
                    newstmt = "{}.{}(lambda row : {} if {} else {}, axis=1)".format(idname, target_name, arg_two, cond, arg_three)
                    return newstmt
                elif target_name == 'astype':
                    newstmt = '({}).{}(( {} ).dtype)'.format(agr_one,target_name,agr_one)
                    return newstmt






