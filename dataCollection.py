import os
from d2api.src import entities
import d2api
import dbutils

api = d2api.APIWrapper(api_key='95F1213EBFC1E3ED35E757F9BD2989F6',parse_response = False)


# filter games with no leavers
def validMatch(matchDetails):
    for player in matchDetails['players']:
        if player["leaver_status"] == 1:
            # TODO print which player is leaver
            return False
    return True


# Hero/Item/Ability information is available without having to specify a key
# print(entities.Hero(67)['hero_name'])
# print(entities.Item(208)['item_aliases'])
# print(entities.Ability(6697)['ability_name'])

if __name__ == '__main__':
    for i in range(127,130):
        data_left_for_hero=True
        last_match_id=0
        while(data_left_for_hero):
            print("fetching matches for hero :",i)
            response=[]
            if last_match_id>0:
                response = api.get_match_history(skill=3, min_players=10, hero_id=i,
                                                 matches_requested=500,start_at_match_id=last_match_id)  # ,start_at_match_id='5126108368' 5126144918
            else:
                response = api.get_match_history(skill=3, min_players=10, hero_id=i,
                                                 matches_requested=500)

            match_history=(d2api.src.util.decode_json(response))['result']
            print("found matches : ",len(match_history['matches']))
            if(len(match_history['matches'])==0):
                data_left_for_hero=False
            for match in match_history['matches']:
                response = api.get_match_details(match['match_id'])
                matchDetails = d2api.src.util.decode_json(response)['result']
                if (validMatch(matchDetails)):
                    dbutils.insertMatchDetail(matchDetails)
                    print("matchAdded!!",matchDetails["match_id"])
                else:
                    print("invalid match!!",matchDetails["match_id"])
                last_match_id=matchDetails["match_id"]+1

