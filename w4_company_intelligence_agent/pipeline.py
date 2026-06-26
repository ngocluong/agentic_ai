import logging
from w4_company_intelligence_agent.summarizer import get_summary
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    website = input("What company are you researching? (enter 'quit' to stop) ")
    if website.lower() == "quit":
        print("Exiting the program.")
    else:
        # Process the website input here
        start = time.time()
        w1 = get_summary(website)
        logger.info(f"Processed {w1['company_name']} in {time.time() - start:.2f}s")
        with open("company_profile.json", "w") as f:
            json.dump(w1, f, indent=2)
        compare = input("Do you want to research another company and compare? (yes/no) ")
        if compare.lower() == "yes":
            website2 = input("Enter the second company website: ")
            start = time.time()
            w2 = get_summary(website2)
            logger.info(f"Processed {w2['company_name']} in {time.time() - start:.2f}s")
            
            with open("company_profile.json", "a") as f:
                json.dump(w2, f, indent=2)

            # Compare the two summaries
            print("\n=== Comparison of Company Summaries ===\n")
            print(f"{'Field':<20} {'Company 1':<50} {'Company 2':<50}")
            print("-" * 120)
            for key in ["company_name", "products", "target_market", "tech_stack", "summary"]:
                val1 = str(w1.get(key, "N/A"))[:48]
                val2 = str(w2.get(key, "N/A"))[:48]
                print(f"{key:<20} {val1:<50} {val2:<50}")
        else:
            print("\n=== Company Summaries ===\n")
            for key in ["company_name", "products", "target_market", "tech_stack", "summary"]:
                val = str(w1.get(key, "N/A"))[:48]
                print(f"{key:<20} {val:<50}")
        
        print("\nSaved results to company_profile.json")