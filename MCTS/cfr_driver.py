from cfr import (train_cfr, save_regret_table,
                 load_regret_table, CFRPolicyBot,
                 evaluate_current_strategy)

if __name__ == "__main__":
    train_cfr(num_iters=15)
    save_regret_table("cfr_table.pkl")