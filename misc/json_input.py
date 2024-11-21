'''
Structure for inputting parameters for strategies
Allows any parameters to be input for a given strategy



'''
from os import name, system


def clear_screen():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def get_input(last_option, prompt_string='Your selection: ', add_back_option = True):
    selection = 0
    if add_back_option:
        last_option += 1
        print(str(last_option), '- Go back')
    while 1:
        selection = input(prompt_string)
        try:
            selection = int(selection)
        except ValueError:
            print('Invalid selection - please try again')
            continue
        if selection < 0 or selection > last_option:
            print('Invalid selection - please try again')
        else:
            break
    if add_back_option and selection == last_option:
        selection = 0
    return selection
    
def get_yes_no(default=0, prompt_string='Your selection (y/n): '):
    # 0 = No
    # 1 = Yes
    selection = input(prompt_string)
    if default == 0:
        return selection.upper() == 'Y'
    else:
        return selection.upper() != 'N'
        
def get_string(prompt_string='Enter a string: '):
    selection = input(prompt_string)
    return selection
    
def get_category(options, prompt_string='Your selection: ', add_back_option=False, return_name=False):
    if len(options) == 0:
        return 0
    elif len(options) == 1:
        return 1
    for i, option in enumerate(options):
        print(i + 1, '-', option)
    selection = get_input(len(options), prompt_string=prompt_string, add_back_option=add_back_option)
    if return_name:
        return options[selection - 1] if selection > 0 else ''
    return selection

class menu_prompts:
    def __init__(self, json_data, root_id=1, memory={}, object_context=None):
        self.data = json_data
        self.root = root_id
        self.mem = memory
        self.context = object_context
        # If the function could not be found in the object context, try to find it here
        self.local_context = {
            'get_category':get_category,
            'get_string':get_string,
            'get_yes_no':get_yes_no,
            'get_input':get_input
        }
        
    def exists(self, data, key):
        return key in data and data[key] != None
        
    def find_id(self, id):
        for obj in self.data:
            if self.exists(obj, 'id') and obj['id'] == id:
                return obj
        return None
        
    def get_name(self, data):
        if self.exists(data, 'name'):
            return data['name']
        return ''
        
    def extract_args(self, args):
        new_args = {}
        for name, value in args.items():
            if len(name) > 0:
                if len(value) > 0 and value[0] == '~':
                    new_value = value[1:]
                    new_args[name] = self.mem[new_value] if self.exists(self.mem, new_value) else None
                elif len(value) > 0 and value[0] == '\\':
                    new_value = value[1:]
                    new_args[name] = new_value
                else:
                    new_args[name] = value
        return new_args
            
        
    def run_functs(self, functs):
        for funct in functs:
            funct_name = self.get_name(funct)
            args = funct['args'] if self.exists(funct, 'args') else {}
            # Make sure the args exist in the context
            new_args = self.extract_args(args)
            
            if hasattr(self.context, funct_name):
                print('running', funct_name, 'with params', new_args)
                ret = getattr(self.context, funct_name)(**new_args)
                print('returned', ret)
                input()
            elif self.exists(self.local_context, funct_name):
                print('running', funct_name, 'with params', new_args)
                ret = self.local_context[funct_name](**new_args)
                print('returned', ret)
                input()
            else:
                return
            if self.exists(funct, 'save_result'):
                print('saved into', funct['save_result'])
                self.mem[funct['save_result']] = ret
            elif ret == 0 or ret == []:
                return
        
    def run(self, id=-1):
        if id == -1:
            id = self.root
        current = self.find_id(id)

        if self.exists(current, 'set'):
            for arg, value in current['set'].items():
                self.mem[arg] = value
        if self.exists(current, 'name'):
            if self.exists(current, 'options'):
                while 1:
                    clear_screen()
                    print(current['name'])
                    # Get all the options
                    option_ids = current['options']
                    options = [self.get_name(self.find_id(option_id)) for option_id in option_ids]
                    choice = get_category(options, add_back_option = True)
                    if choice == 0:
                        return
                    # Recurse
                    self.run(option_ids[choice - 1])
            elif self.exists(current, 'execute'):
                # Run a function
                run_functs = current['execute']
                self.run_functs(run_functs)
            return


class input_generic:
    def __init__(self, id, name, type):
        self.id = id
        self.name = name
        self.type = type
    
    def prompt(self, level=0):
        clear_screen()
        print(self.name, '-', self.type)
        if level == 0:
            print('Options:')
            if self.id != 0:
                print('1 - Edit')
                print('2 - Delete')
        elif level == 1:
            print('What would you like to edit?')
            print('1 - name (' + self.name + ')')
            print('2 - type (' + self.type + ')')
        elif level == 2:
            print('Enter a new name for this object:')
        elif level == 3:
            print('Enter a new type for this object:')
        elif level == 4:
            print('Are you sure you want to delete this object?')
            
    def take_action(self, level=0):
        self.prompt(level)
        if level == 0:      # Entry
            choice = get_input(2)
            if choice == 0:
                return 0
            if choice == 1:
                self.take_action(1)
            elif choice == 2:
                self.take_action(4)
        elif level == 1:    # Edit
            choice = get_input(2)
            if choice == 0:
                self.take_action(0)
            elif choice == 1:
                self.take_action(2)
            elif choice == 2:
                self.take_action(3)
        elif level == 2:    # Edit name
            choice = get_string()
            if choice == "":
                self.take_action(1)
            else:
                self.name = choice
        elif level == 3:    # Edit type
            choice = get_category(get_all_types())
            if choice == 0:
                self.take_action(1)
            else:
                return choice
        elif level == 4:    # Delete
            choice = get_yes_no()
            if choice == 0:
                self.take_action(0)
            else:
                return -1
            
            
    def get_all_types():
        return ['object', 'list', 'int', 'float', 'string']
    
    def get_dict(self):
        dict = {'id':self.id,
                'name':self.name,
                'type':self.type}
        return dict
    
    def __repr__(self):
        dict = self.get_dict()
        return str(dict)

class input_object(input_generic):
    def __init__(self, id, name, value=None):
        super().__init__(id, name, 'object')
        self.type = 'object'
        self.value = value
    
    def get_dict(self):
        dict = super().get_dict()
        dict['value'] = self.value
        return dict
    
class input_list(input_generic):
    def __init__(self, id, name, elements):
        super().__init__(id, name, 'list')
        self.elements = elements
        
    def get_dict(self):
        dict = super().get_dict()
        dict['elements'] = self.elements

class input_int(input_generic):
    def __init__(self, id, name):
        super().__init__(id, name, 'int')
        
class input_float(input_generic):
    def __init__(self, id, name):
        super().__init__(id, name, 'float')
        
class input_string(input_generic):
    def __init__(self, id, name):
        super().__init__(id, name, 'string')
        
def get_inputs(input_json):
    # Text-based prompts to get values for inputs
    pass

def find_input(inputs, id):
    for input in inputs:
        if input.id == id:
            return input
    return None
    
def create_inputs():
    # Init variables
    current_id = 0
    selected_id = 0
    
    # Make an object first to hold the inputs
    root = input_object(current_id, 'parameter', 'parameter')
    elements = [root]
    selected_input = root
    
    # Loop until the user is done creating inputs
    while 1:
        print(root)
        # Prompt the user to select an input element
        while selected_id < 0:
            print(root)
            selected_id = input('Input which element to change: ')
            # Find the corresponding element
            selected_input = find_input(inputs, selected_id)
            if selected_input is None:
                print('This input element does not exist')
                selected_id = -1
        
        # Prompt for the input's options
        selected_input.take_action()

if __name__ == "__main__":
    create_inputs()