import ultrametric_distance
#from ultrametric_distance import get_ultrametric_distance

def main():

    print("printing available functions in ultrametric_distance:")
    print(dir(ultrametric_distance))

    tree1 = {
        "entity": {
            "physical_entity": {
                "object": {
                    "whole": {
                        "plant": {
                            "legume": {
                                "bean1": {}
                            }
                        }
                    }
                }
            }
        }
    }

    tree2 = {
        "entity": {
            "physical_entity": {
                "object": {
                    "whole": {
                        "plant": {
                            "legume": {
                                "bean1": {}
                            }
                        }
                    }
                },
                "food": {
                    "plant_product": {
                        "vegetable": {
                            "bean2": {}
                        }
                    }
                }
            }
        }
    }

    distance = ultrametric_distance.get_ultrametric_distance(tree1, tree2)
    print("Ultrametric distance:", distance)

if __name__ == "__main__":
    main()