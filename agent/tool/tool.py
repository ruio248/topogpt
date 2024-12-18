import os
import json
import requests

# Tool Functions

# Add description information for the tool
# Implement specific requests for crystal generation models


class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()
    
    def _tools(self):
        tools = [
            {
                'name_for_human': 'crystal generation model',
                'name_for_model': 'Con-CDVAE_topo',
                'description_for_model': 'The generation of crystals can be based on topological classification and other customizable parameters like band gap, formation energy, atom count, etc.',
                'parameters': [
                    {
                        'name': 'n_cry',
                        'description': 'The number of crystals to generate (1-200). If not set, defaults to "None" which is treated as 1.',
                        'required': False,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'bg',
                        'description': 'Desired band gap (0-7 eV) or "None" if not restricted.',
                        'required': False,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'fe',
                        'description': 'Desired formation energy (-5 to 0.5 eV/atom) or "None" if not restricted.',
                        'required': False,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'n_atom',
                        'description': 'Number of atoms in the unit cell (1-20) or "None" to let the program set it randomly.',
                        'required': False,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'formula',
                        'description': 'Chemical formula of the crystal. If not set, defaults to "None".',
                        'required': False,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'topo_class',
                        'description': 'Topological classification: "Triv_Ins", "HSP_SM", "HSL_SM", "TI", "TCI", or "None".',
                        'required': False,
                        'schema': {'type': 'string'},
                    }
                ],
            }
        ]
        return tools

    def crystal_generation(self, n_cry: str = "None", bg: str = "None", fe: str = "None", n_atom: str = "None", formula: str = "None", topo_class: str = "None"):
        """
        Parameters:
        - n_cry: Integer from 1 to 200 or 'None' (defaults to 1 if not set)
        - bg: Float from 0 to 7 eV or 'None'
        - fe: Float from -5 to 0.5 eV/atom or 'None'
        - n_atom: Integer from 1 to 20 or 'None' (random if 'None')
        - formula: Chemical formula string or 'None'
        - topo_class: Topological class ('Triv_Ins', 'HSP_SM', 'HSL_SM', 'TI', 'TCI', or 'None')
        """
        
        # Build the request URL dynamically based on the provided parameters
        url = f"http://172.16.8.34:8083/fulltopo/{n_cry}/{bg}/{fe}/{n_atom}/{formula}/{topo_class}"
        
        try:
            # Make the GET request
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()  # Assuming the response is in JSON format
            else:
                return {"error": f"Request failed with status code {response.status_code}"}
        
        except Exception as e:
            return {"error": str(e)}
