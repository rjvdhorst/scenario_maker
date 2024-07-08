# from load_profiles import LoadProfile

# class GridConnections:
#     def __init__(self, substation, loads):
#         self.substation = substation
#         self.loads = loads # Should be a list of Load objects

#     def add_load(self, load):
#         self.loads.append(load)

#     def remove_load(self, load):
#         self.loads.remove(load)

#     def get_total_load(self):
#         total_load_profile = {}
#         for load in self.loads:
#             load_profile = load.get_profile()
#             for timestep in load_profile['time_array']:
#                 if timestep['time'] not in total_load_profile:
#                     total_load_profile[timestep['time']] = 0
#                 total_load_profile[timestep['time']] += timestep['value']
        
#         total_load_profile['name'] = self.substation

    
