import os
import json
import csv
from pathlib import Path
from contextgem import Document, DocumentLLM, StringConcept

class ContextGemQAGenerator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Set for ContextGem
        os.environ["OPENAI_API_KEY"] = self.api_key
        print("✅ ContextGem QA Generator initialized")
    
    def load_all_context_files(self, context_dir):
        """Load and combine all context files into a single text block"""
        combined_text = []
        processed_files = []
        
        for file_path in Path(context_dir).glob("*"):
            if file_path.is_file():
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                    
                    if content:
                        combined_text.append(f"=== {file_path.name} ===\n{content}")
                        processed_files.append(file_path.name)
                        print(f"✅ Loaded: {file_path.name}")
                    
                except Exception as e:
                    print(f"⚠️ Could not read {file_path.name}: {e}")
        
        print(f"\n📊 Successfully loaded {len(processed_files)} files")
        return "\n\n".join(combined_text)
    
    def generate_qa_pairs(self, combined_context, num_pairs=10):
        """Generate Q&A pairs using ContextGem, handling large batches"""
        
        all_pairs = []
        
        # For large requests, break into smaller batches
        if num_pairs > 25:
            batch_size = 25
            batches = (num_pairs + batch_size - 1) // batch_size  # Ceiling division
            print(f"🔄 Large request detected. Breaking into {batches} batches of {batch_size} pairs each...")
            
            for batch_num in range(batches):
                batch_pairs = min(batch_size, num_pairs - len(all_pairs))
                print(f"📦 Processing batch {batch_num + 1}/{batches} ({batch_pairs} pairs)...")
                
                batch_result = self._generate_single_batch(combined_context, batch_pairs)
                all_pairs.extend(batch_result)
                
                if len(all_pairs) >= num_pairs:
                    break
            
            print(f"✅ Completed all batches. Generated {len(all_pairs)} total pairs")
            return all_pairs[:num_pairs]
        else:
            return self._generate_single_batch(combined_context, num_pairs)
    
    def _generate_single_batch(self, combined_context, batch_pairs):
        """Generate a single batch using ContextGem"""
        try:
            print(f"🔄 Using ContextGem to generate {batch_pairs} pairs...")
            
            # Create document with context
            doc = Document(raw_text=combined_context)
            
            # Create the QA concept with better formatting
            qa_concept = StringConcept(
                name="QA_Dataset",
                description=(
                    f"Generate exactly {batch_pairs} question-answer pairs for fine-tuning. "
                    f"Use this EXACT format with clear line breaks:\n\n"
                    f"Q1: [question text]\n"
                    f"A1: [answer text]\n\n"
                    f"Q2: [question text]\n"
                    f"A2: [answer text]\n\n"
                    f"Continue for all {batch_pairs} pairs. Create diverse questions covering "
                    f"calculations, definitions, processes, relationships, and problem-solving. "
                    f"Make answers detailed with specific references."
                )
            )
            
            # Add concept to document
            doc.add_concepts([qa_concept])
            
            # Create LLM for GPT-5
            llm = DocumentLLM(
                model="openai/gpt-5",
                api_key=self.api_key
            )
            
            # Extract using ContextGem
            doc = llm.extract_all(doc)
            
            # Get the extracted content
            extracted_concept = doc.get_concept_by_name("QA_Dataset")
            if extracted_concept and extracted_concept.extracted_items:
                # Get the first extracted item's value
                content = extracted_concept.extracted_items[0].value
                
                print(f"\n🔍 DEBUG - ContextGem generated content length: {len(content) if content else 0} characters")
                
                parsed_pairs = self._robust_parse_qa(content, batch_pairs)
                print(f"✅ ContextGem batch generated {len(parsed_pairs)} Q&A pairs")
                return parsed_pairs
            else:
                print("⚠️ No extracted items from ContextGem")
                return []
                
        except Exception as e:
            print(f"❌ ContextGem batch failed: {e}")
            return []
    
    def _robust_parse_qa(self, content, expected_pairs):
        """Robust parsing that handles ContextGem's various output formats"""
        qa_pairs = []
        
        if not content:
            print("⚠️ No content to parse")
            return qa_pairs
        
        import re
        
        # Method 1: Try to find Q/A patterns with regex
        # This handles cases where everything is on one line or properly formatted
        pattern = r'Q(\d+):\s*(.*?)\s*A\1:\s*(.*?)(?=\s*Q\d+:|$)'
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            question_num, question, answer = match
            question = question.strip()
            answer = answer.strip()
            
            if question and answer:
                qa_pairs.append({
                    "prompt": question,
                    "completion": answer
                })
        
        print(f"📝 Method 1 (regex) found {len(qa_pairs)} pairs")
        
        # Method 2: If regex didn't work well, try line-by-line parsing
        if len(qa_pairs) < expected_pairs * 0.8:  # Less than 80% of expected
            print("🔄 Trying alternative parsing method...")
            
            lines = content.split('\n')
            current_q = None
            current_a = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for Q pattern
                q_match = re.match(r'Q\d+:\s*(.*)', line, re.IGNORECASE)
                if q_match:
                    # Save previous pair if we have both Q and A
                    if current_q and current_a:
                        qa_pairs.append({
                            "prompt": current_q,
                            "completion": current_a
                        })
                    current_q = q_match.group(1).strip()
                    current_a = None
                    continue
                
                # Look for A pattern
                a_match = re.match(r'A\d+:\s*(.*)', line, re.IGNORECASE)
                if a_match:
                    current_a = a_match.group(1).strip()
                    continue
                
                # If we're building an answer, append to it
                if current_q and current_a is not None:
                    current_a += " " + line
                elif current_q and not current_a:
                    # This might be continuation of question or start of answer
                    if line.startswith(('A', 'Answer')):
                        current_a = line
                    else:
                        current_q += " " + line
            
            # Don't forget the last pair
            if current_q and current_a:
                qa_pairs.append({
                    "prompt": current_q,
                    "completion": current_a
                })
            
            print(f"📝 Method 2 (line-by-line) total found {len(qa_pairs)} pairs")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_pairs = []
        for pair in qa_pairs:
            pair_key = (pair['prompt'][:50], pair['completion'][:50])  # Use first 50 chars as key
            if pair_key not in seen:
                seen.add(pair_key)
                unique_pairs.append(pair)
        
        print(f"📝 Final count after deduplication: {len(unique_pairs)} pairs")
        return unique_pairs[:expected_pairs]
    
    def save_to_csv(self, qa_pairs, filename="qa_dataset.csv"):
        """Save to CSV for fine-tuning"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['prompt', 'completion'])
            for pair in qa_pairs:
                writer.writerow([pair['prompt'], pair['completion']])
        print(f"💾 Saved {len(qa_pairs)} pairs to {filename}")
    
    def save_to_json(self, qa_pairs, filename="qa_dataset.json"):
        """Save to JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(qa_pairs)} pairs to {filename}")

def main():
    CONTEXT_DIR = "/Users/sanjanamanikandan/Downloads/context_files"
    
    try:
        generator = ContextGemQAGenerator()
        
        num_pairs = int(input("How many Q/A pairs do you want to generate? "))
        
        print("🔄 Loading context files...")
        combined_context = generator.load_all_context_files(CONTEXT_DIR)
        
        if not combined_context:
            print("❌ No context found")
            return
        
        print(f"📄 Combined context length: {len(combined_context)} characters")
        
        qa_pairs = generator.generate_qa_pairs(combined_context, num_pairs)
        
        if qa_pairs:
            generator.save_to_csv(qa_pairs)
            generator.save_to_json(qa_pairs)
            
            print(f"\n📋 Sample of generated pairs:")
            for i, pair in enumerate(qa_pairs[:3], 1):
                print(f"{i}. Q: {pair['prompt']}")
                print(f"   A: {pair['completion'][:150]}...")
                print()
            
            print(f"✅ Successfully generated {len(qa_pairs)} pairs!")
            
            if len(qa_pairs) < num_pairs:
                print(f"⚠️ Generated {len(qa_pairs)} pairs instead of {num_pairs}. This may be due to ContextGem limitations or parsing issues.")
        else:
            print("❌ No pairs generated")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
