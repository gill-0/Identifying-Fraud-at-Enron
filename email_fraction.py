
def computeFraction( poi_messages, all_messages ):
    if poi_messages == 'NaN' or all_messages == "NaN":
        fraction = 0
    else:
        fraction = (float(poi_messages) / all_messages)

    return fraction


def email_poi_fraction(data_dict):
    for name in data_dict:
        data_point = data_dict[name]

        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
        data_point["fraction_from_poi"] = fraction_from_poi

        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages)
        data_point["fraction_to_poi"] = fraction_to_poi



