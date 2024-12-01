import os
import json
import argparse
import openai
import typing

def load_arc_tasks(tasks_path: str, solutions_path: str = None) -> typing.List[dict]:
    """
    Load ARC tasks from JSON files
    
    Args:
        tasks_path (str): Path to the tasks JSON file
        solutions_path (str, optional): Path to the solutions JSON file
    
    Returns:
        list: List of ARC task dictionaries
    """
    # Load tasks
    with open(tasks_path, 'r') as f:
        tasks_data = json.load(f)

    # Load solutions if provided
    solutions = {}
    if solutions_path:
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)
    
    # Prepare tasks list
    processed_tasks = []
    for task_id, task_info in tasks_data.items():
        # Prepare task dictionary
        task_dict = {
            'task_id': task_id,
            'train': task_info.get('train', []),
            'test': task_info.get('test', []),
        }
        
        # Add solution if available
        if task_id in solutions:
            task_dict['solution'] = solutions[task_id]
        
        processed_tasks.append(task_dict)
    
    return processed_tasks

class ARCSolver:
    def __init__(self, model_name='llama31-405b-fp8', base_url=None, api_key=None):
        """
        Initialize the ARC solver with model configuration
        
        Args:
            model_name (str): Name of the model to use
            base_url (str): Base URL for the API
            api_key (str): API key for authentication
        """
        self.client = openai.OpenAI(
            api_key=api_key or "cmsc-35360",
            base_url=base_url or "http://66.55.67.65:80/v1"
        )
        self.model_name = model_name

    def generate_solution(self, task, temperature):
        """
        Generate a solution for an ARC task with explicit delimiters
        
        Args:
            task (dict): ARC task dictionary
        
        Returns:
            dict: Solution generation response
        """
        # Construct prompt with training examples
        train_examples = self.format_train_examples(task.get('train', []))
        test_input = self.grid_to_string(task['test'][0]['input'])

        prompt = f"""ARC Task Solving Instructions:
You will be given multiple training examples and a test input grid. Your goal is to apply the transformation rule from the training examples to the test input grid.

Training Examples:
{train_examples}

Test Input Grid:
{test_input}

Please generate the output grid that follows the underlying transformation rule. 
IMPORTANT: 

1. Provide a clear rationale explaining how you will derive the solution
2. Wrap your solution grid in triple backticks (```) like this:
```
solution grid here
```"""  
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at solving Abstract and Reasoning Challenge (ARC) tasks. Carefully analyze the input grids and generate the correct output grid. Always wrap your solution in triple backticks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2048
            )
            
            return {
                'task_id': task.get('task_id', 'unknown'),
                'input': f"""Training Examples: 
{train_examples}

Test Input Grid:
{test_input}""",
                'generated_solution': response.choices[0].message.content,
                'raw_response': response
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'task_id': task.get('task_id', 'unknown')
            }
    
    def format_train_examples(self, train_examples):
        """
        Format training examples for the prompt
        
        Args:
            train_examples (list): List of training examples
        
        Returns:
            str: Formatted training examples
        """
        formatted_examples = []
        for example in train_examples:
            input_grid = self.grid_to_string(example['input'])
            output_grid = self.grid_to_string(example['output'])
            formatted_examples.append(f"Input Grid:\n{input_grid}\n\nOutput Grid:\n{output_grid}\n---")
        
        return "\n".join(formatted_examples)
    
    def grid_to_string(self, grid):
        """
        Convert a 2D grid to a readable string representation
        
        Args:
            grid (list): 2D grid of integers
        
        Returns:
            str: String representation of the grid
        """
        return '\n'.join([' '.join(map(str, row)) for row in grid])
    
    def parse_solution(self, solution_str):
        """
        Parse a solution string into a 2D grid using code block delimiters
        
        Args:
            solution_str (str): String representation of solution
        
        Returns:
            list: 2D grid parsed from solution
        """
        try:
            # Extract content between triple backticks
            import re
            
            # Find all code blocks
            code_blocks = re.findall(r'```(.*?)```', solution_str, re.DOTALL)
            
            # If no code blocks found, return None
            if not code_blocks:
                print("No code blocks found in solution")
                return None

            # Use the last block
            block = code_blocks[-1].strip()
            
            # Split into lines and parse
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            
            # Parse each line into integers
            grid = [
                [int(x) for x in line.split()] 
                for line in lines
            ]
            
            return grid
        
        except Exception as e:
            print(f"Error parsing solution: {e}")
            print(f"Problematic solution string: {solution_str}")
            return None
    
    def evaluate_solution(self, task, solution_str):
        """
        Evaluate the generated solution against the true output
        
        Args:
            task (dict): Original ARC task
            solution_str (str): Generated solution string
        
        Returns:
            bool: Whether the solution matches the true output
        """
        # Parse the solution

        # print(f"solution str: {solution_str} \n")

        parsed_solution = self.parse_solution(solution_str)

        # print(f"parsed_solution: {parsed_solution} \n")
        
        # If no solution parsed, return False
        if parsed_solution is None:
            return False

        # If task has a predefined solution, compare with it
        if 'solution' in task:
            # Assuming solution is the first (and only) item in the list
            true_solution = task['solution'][0]
            
            # Compare parsed solution with true solution

            # print (f"true_solution: {true_solution} \n")
            return parsed_solution == true_solution
        
        return False
    
    def solve_arc_dataset(self, tasks_path, solutions_path=None):
        """
        Solve entire ARC dataset and save results
        
        Args:
            tasks_path (str): Path to ARC tasks JSON
            solutions_path (str, optional): Path to solutions JSON
            output_path (str): Path to save results JSON
        """
        # Load ARC tasks
        tasks = load_arc_tasks(tasks_path, solutions_path)

        for temperature in [0.4, 0.6, 0.8, 1]:
            results = []
            i = 0
            for task in tasks:
            
                # Generate solution
                solution_response = self.generate_solution(task, temperature)
                
                # Evaluate solution if possible
                is_correct = False
                if 'generated_solution' in solution_response:
                    is_correct = self.evaluate_solution(
                        task, 
                        solution_response['generated_solution']
                    )

                result = {
                    'task': solution_response.get('task', ''),
                    'input': solution_response.get('input', ''),
                    'output': solution_response.get('generated_solution', ''),
                    'label': 1 if is_correct else 0,
                }
                
                results.append(result)
                i += 1

                if i % 10 == 0:
                    print(f"{i} / {len(tasks)} complete for temperature {temperature}")

            # Save results to JSON
            with open(f"arc_results_{temperature}.json", 'w') as f:
                json.dump(results, f, indent=2)

            print(f"Results saved to arc_results_{temperature}.json")

def main():
    parser = argparse.ArgumentParser(description="ARC Challenge Solver")
    parser.add_argument("--tasks", type=str, default = "data/arc-agi_training_challenges.json", 
                        help="Path to ARC tasks JSON")
    parser.add_argument("--solutions", type=str, default = "data/arc-agi_training_solutions.json",
                        help="Path to ARC solutions JSON")
    parser.add_argument("--port", type=int, default=80, 
                        help="API port number")
    
    args = parser.parse_args()
    
    # Configure API endpoint
    base_url = f"http://66.55.67.65:{args.port}/v1"
    
    # Initialize solver
    solver = ARCSolver(base_url=base_url)
    
    # Solve dataset
    solver.solve_arc_dataset(
        tasks_path=args.tasks, 
        solutions_path=args.solutions,
    )

if __name__ == "__main__":
    main()