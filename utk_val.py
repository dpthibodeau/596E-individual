import numpy as np


def main():
    res_file = "UTKFACE_out.npy"
    res_arr = np.load(res_file, allow_pickle=True)
    child_count = adult_count = child_correct = adult_correct = 0
    gender_correct = 0
    no_predict_count = 0
    print(f"Total images: {len(res_arr)}")
    for r in res_arr:
        res = r["result"][0]
        if "no_predict" in res:
            no_predict_count += 1
            continue
        pred_age = res["label"]
        pred_gender = res["gender"]
        true_label = r["file_path"].split("\\")[-1].split("_")
        t_age, t_gender = int(true_label[0]), ("male" if true_label[1] == "0" else "female")
        if t_age < 18:
            child_count += 1
            if pred_age == "child":
                child_correct += 1
        else:
            adult_count += 1
            if pred_age == "adult":
                adult_correct += 1

        if t_gender == pred_gender:
            gender_correct += 1

    print(f"Gender accuracy: {gender_correct/len(res_arr)}")
    print(f"Child accuracy: {child_correct/child_count}")
    print(f"Adult accuracy: {adult_correct/adult_count}")
    print(f"No prediction percentage: {no_predict_count/len(res_arr)}")
    print(f"Total accuracy: {(child_correct+adult_correct)/(child_count+adult_count)}")


if __name__ == "__main__":
    main()
