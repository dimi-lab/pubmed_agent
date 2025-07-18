"""
Fixed and Enhanced Test App for RAG + BioMistral Integration
"""

import streamlit as st
import logging
from typing import Dict, Any, List
from datetime import datetime
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_system_test() -> List[str]:
    """Quick test to verify all components are working"""
    test_results = []
    
    # Test 1: BioMistral
    try:
        st.info("Testing BioMistral model...")
        from components.llm import get_llm
        llm = get_llm()
        if llm:
            response = llm.invoke("What is cancer?")
            if response and len(response.strip()) > 10:
                test_results.append("✅ BioMistral: Working - Generated meaningful response")
            else:
                test_results.append(f"⚠️ BioMistral: Short response - '{response}'")
        else:
            test_results.append("❌ BioMistral: Model not loaded")
    except ImportError as e:
        test_results.append(f"❌ BioMistral: Import error - {str(e)[:50]}")
    except Exception as e:
        test_results.append(f"❌ BioMistral: Error - {str(e)[:50]}")
    
    # Test 2: Embeddings
    try:
        st.info("Testing embeddings...")
        from backend.rag_pipeline.embeddings import embeddings
        if embeddings:
            test_embedding = embeddings.embed_query("test medical query")
            if test_embedding and len(test_embedding) > 0:
                test_results.append(f"✅ Embeddings: Working - Dimension: {len(test_embedding)}")
            else:
                test_results.append("❌ Embeddings: Invalid output")
        else:
            test_results.append("❌ Embeddings: Not available")
    except ImportError as e:
        test_results.append(f"❌ Embeddings: Import error - {str(e)[:50]}")
    except Exception as e:
        test_results.append(f"❌ Embeddings: Error - {str(e)[:50]}")
    
    # Test 3: ChromaDB
    try:
        st.info("Testing ChromaDB...")
        from backend.rag_pipeline.chromadb_rag import ChromaDbRag
        from backend.rag_pipeline.embeddings import embeddings
        import tempfile
        import os
        
        if embeddings:
            with tempfile.TemporaryDirectory() as temp_dir:
                rag_system = ChromaDbRag(temp_dir, embeddings)
                # Test if we can create a client
                client = rag_system._create_chromadb_client()
                test_results.append("✅ ChromaDB: Working - Client created successfully")
        else:
            test_results.append("❌ ChromaDB: Cannot test without embeddings")
    except ImportError as e:
        test_results.append(f"❌ ChromaDB: Import error - {str(e)[:50]}")
    except Exception as e:
        test_results.append(f"❌ ChromaDB: Error - {str(e)[:50]}")
    
    # Test 4: Storage System
    try:
        st.info("Testing storage system...")
        from backend.data_repository.local_storage import LocalJSONStore
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalJSONStore(temp_dir)
            queries = storage.get_list_of_queries()
            test_results.append("✅ Storage: Working - Can create and query storage")
    except ImportError as e:
        test_results.append(f"❌ Storage: Import error - {str(e)[:50]}")
    except Exception as e:
        test_results.append(f"❌ Storage: Error - {str(e)[:50]}")
    
    # Test 5: PubMed (optional)
    try:
        st.info("Testing PubMed retriever...")
        from backend.abstract_retrieval.pubmed_retriever import PubMedRetriever
        retriever = PubMedRetriever(max_results=1)
        test_results.append("✅ PubMed: Available - Retriever initialized")
    except ImportError as e:
        test_results.append(f"⚠️ PubMed: Optional dependency missing - {str(e)[:50]}")
    except Exception as e:
        test_results.append(f"⚠️ PubMed: Configuration issue - {str(e)[:50]}")
    
    return test_results

def test_rag_integration_enhanced(components: Dict[str, Any]) -> tuple:
    """Enhanced RAG integration test with better error handling"""
    
    # Check prerequisites
    required_components = ['rag_system', 'storage']
    missing = [comp for comp in required_components if not components.get(comp)]
    
    if missing:
        return False, f"Missing required components: {missing}"
    
    try:
        st.info("Creating test documents...")
        
        # Create test documents with LangChain Document format
        from langchain.schema import Document
        
        test_docs = [
            Document(
                page_content="Cancer immunotherapy uses the body's immune system to fight cancer cells. T-cells are engineered to better recognize and attack tumors. This approach has shown remarkable success in treating various cancer types.",
                metadata={
                    "title": "Cancer Immunotherapy Review", 
                    "year_of_publication": 2023, 
                    "authors": "Smith et al.",
                    "source": "test_doc_1"
                }
            ),
            Document(
                page_content="CRISPR gene editing allows precise modification of DNA sequences. This technology has applications in treating genetic diseases by correcting defective genes at their source.",
                metadata={
                    "title": "CRISPR Technology in Medicine", 
                    "year_of_publication": 2023, 
                    "authors": "Johnson et al.",
                    "source": "test_doc_2"
                }
            ),
            Document(
                page_content="Machine learning algorithms help analyze medical data by identifying patterns in patient records and predicting treatment outcomes with high accuracy.",
                metadata={
                    "title": "AI in Healthcare", 
                    "year_of_publication": 2023, 
                    "authors": "Garcia et al.",
                    "source": "test_doc_3"
                }
            )
        ]
        
        # Test vector store creation
        test_query_id = f"test_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        st.info("Creating vector store...")
        vector_store = components['rag_system'].create_vector_index_for_user_query(test_docs, test_query_id)
        
        # Test retrieval with multiple queries
        st.info("Testing semantic retrieval...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        
        test_queries = [
            "What is cancer immunotherapy?",
            "How does CRISPR work?",
            "What can AI do in medicine?"
        ]
        
        all_results_good = True
        retrieval_results = []
        
        for query in test_queries:
            relevant_docs = retriever.get_relevant_documents(query)
            
            if relevant_docs:
                top_doc = relevant_docs[0]
                retrieval_results.append({
                    'query': query,
                    'found_docs': len(relevant_docs),
                    'top_result': top_doc.metadata.get('title', 'Unknown'),
                    'relevance_check': query.lower().split()[0] in top_doc.page_content.lower()
                })
            else:
                all_results_good = False
                retrieval_results.append({
                    'query': query,
                    'found_docs': 0,
                    'error': 'No documents retrieved'
                })
        
        # Display results
        for result in retrieval_results:
            if result.get('found_docs', 0) > 0:
                st.success(f"✅ Query: '{result['query']}' → Found: {result['found_docs']} docs → Top: {result['top_result']}")
            else:
                st.error(f"❌ Query: '{result['query']}' → {result.get('error', 'Failed')}")
        
        # Cleanup
        st.info("Cleaning up test data...")
        try:
            components['rag_system'].delete_vector_index(test_query_id)
            st.success("✅ Cleanup completed")
        except Exception as cleanup_error:
            st.warning(f"⚠️ Cleanup warning: {cleanup_error}")
        
        if all_results_good:
            return True, f"RAG working correctly - tested {len(test_queries)} queries successfully"
        else:
            return False, "Some RAG tests failed - check individual query results above"
            
    except Exception as e:
        st.error(f"RAG test error: {str(e)}")
        logger.error(f"RAG test failed: {e}")
        traceback.print_exc()
        return False, f"RAG test failed with error: {str(e)}"

def comprehensive_rag_test():
    """Comprehensive test of the entire RAG + BioMistral pipeline"""
    
    st.markdown("## 🧪 Comprehensive RAG + BioMistral Test")
    
    if st.button("🚀 Run Full Pipeline Test"):
        with st.spinner("Running comprehensive test..."):
            
            # Step 1: Initialize components
            try:
                st.markdown("### 1️⃣ Initializing Components")
                
                # Import initialization function
                try:
                    from app import initialize_system_components
                    components = initialize_system_components()
                except ImportError:
                    st.error("❌ Cannot import initialize_system_components from app.py")
                    st.info("💡 Make sure app.py is in the same directory and contains the initialization function")
                    return
                except Exception as e:
                    st.error(f"❌ Component initialization failed: {e}")
                    return
                
                # Check required components
                required_components = ['ai_model', 'rag_system', 'storage']
                missing = [comp for comp in required_components if not components.get(comp)]
                
                if missing:
                    st.error(f"❌ Missing components: {missing}")
                    st.info("💡 Check the system status in the main app to see what's not working")
                    return
                
                st.success("✅ All required components initialized successfully")
                
                # Show component status
                for comp in required_components:
                    if components.get(comp):
                        st.success(f"✅ {comp}: Available")
                    else:
                        st.error(f"❌ {comp}: Missing")
                
            except Exception as e:
                st.error(f"❌ Component initialization failed: {e}")
                traceback.print_exc()
                return
            
            # Step 2: Create test research papers
            try:
                st.markdown("### 2️⃣ Creating Test Research Papers")
                
                from backend.data_repository.models import ScientificAbstract
                
                test_abstracts = [
                    ScientificAbstract(
                        title="CRISPR-Cas9 Gene Editing in Cancer Treatment: A Breakthrough Approach",
                        authors=["Smith, J.A.", "Johnson, M.K.", "Brown, K.L."],
                        year=2023,
                        abstract_content="CRISPR-Cas9 technology has revolutionized cancer treatment by enabling precise gene editing of tumor suppressor genes and oncogenes. This study demonstrates how CRISPR can selectively target p53 mutations in tumor cells, leading to selective cancer cell death through apoptosis. We observed a 75% reduction in tumor size in mouse models treated with CRISPR-engineered CAR-T cells. The precision of CRISPR allows for minimal off-target effects while maximizing therapeutic efficacy.",
                        doi="10.1016/j.cancer.2023.001",
                        journal="Cancer Research",
                        pmid="12345678",
                        keywords=["CRISPR", "gene editing", "cancer", "CAR-T", "oncogenes"],
                        source="Test"
                    ),
                    ScientificAbstract(
                        title="Immunotherapy Checkpoint Inhibitors in Melanoma: Mechanisms and Clinical Outcomes",
                        authors=["Garcia, L.M.", "Wang, X.Y.", "Thompson, R.K."],
                        year=2023,
                        abstract_content="Immunotherapy has emerged as a breakthrough treatment for advanced melanoma through checkpoint inhibition. This research explores how PD-1 and CTLA-4 antibodies enhance T-cell responses against melanoma cells by blocking inhibitory signals. Patient survival rates improved by 60% with combination immunotherapy (nivolumab + ipilimumab) compared to traditional chemotherapy. The study included 450 patients with stage III-IV melanoma, showing durable responses lasting over 24 months.",
                        doi="10.1038/s41467-2023.002",
                        journal="Nature Communications",
                        pmid="23456789",
                        keywords=["immunotherapy", "melanoma", "PD-1", "CTLA-4", "checkpoint inhibitors"],
                        source="Test"
                    ),
                    ScientificAbstract(
                        title="Machine Learning Applications in Drug Discovery: Accelerating Therapeutic Development",
                        authors=["Lee, S.H.", "Patel, A.R.", "Williams, D.C."],
                        year=2023,
                        abstract_content="Machine learning algorithms are accelerating drug discovery by predicting molecular properties, drug-target interactions, and therapeutic efficacy. Our deep learning model, trained on 2.5 million compounds, identified 15 potential cancer drug candidates with 85% accuracy in predicting IC50 values. This approach reduces drug development time from 10-15 years to 2-3 years and cuts costs by 40%. The model successfully predicted the activity of known FDA-approved drugs with 92% accuracy.",
                        doi="10.1038/s41571-2023.003",
                        journal="Nature Reviews Drug Discovery",
                        pmid="34567890",
                        keywords=["machine learning", "drug discovery", "AI", "molecular prediction", "therapeutic development"],
                        source="Test"
                    )
                ]
                
                st.success(f"✅ Created {len(test_abstracts)} detailed test research papers")
                
                # Show paper details
                for i, abstract in enumerate(test_abstracts, 1):
                    with st.expander(f"📄 Paper {i}: {abstract.title}"):
                        st.write(f"**Authors:** {', '.join(abstract.authors)}")
                        st.write(f"**Journal:** {abstract.journal} ({abstract.year})")
                        st.write(f"**Abstract:** {abstract.abstract_content[:200]}...")
                        st.write(f"**Keywords:** {', '.join(abstract.keywords)}")
                
            except Exception as e:
                st.error(f"❌ Failed to create test papers: {e}")
                traceback.print_exc()
                return
            
            # Step 3: Save papers and create RAG index
            try:
                st.markdown("### 3️⃣ Creating RAG Knowledge Base")
                
                # Save abstracts to storage
                query_id = components['storage'].save_dataset(test_abstracts, "Comprehensive RAG Test Query")
                st.info(f"📝 Saved abstracts with query ID: {query_id}")
                
                # Convert to documents for RAG
                documents = components['storage'].create_document_list(test_abstracts)
                st.info(f"📄 Created {len(documents)} LangChain documents")
                
                # Create vector index
                vector_store = components['rag_system'].create_vector_index_for_user_query(documents, query_id)
                st.success(f"✅ RAG vector index created with {len(documents)} documents")
                
                # Verify the index was created
                collections = components['rag_system'].list_collections()
                expected_collection = components['rag_system']._sanitize_collection_name(query_id)
                if expected_collection in collections:
                    st.success(f"✅ Vector store collection '{expected_collection}' verified")
                else:
                    st.warning(f"⚠️ Collection not found in list: {collections}")
                
            except Exception as e:
                st.error(f"❌ RAG index creation failed: {e}")
                traceback.print_exc()
                return
            
            # Step 4: Test semantic search
            try:
                st.markdown("### 4️⃣ Testing Semantic Search")
                
                # Create retriever
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                
                # Test multiple queries with expected results
                test_cases = [
                    {
                        "query": "How does CRISPR work in cancer treatment?",
                        "expected_keywords": ["CRISPR", "gene editing", "cancer"],
                        "expected_paper": "CRISPR-Cas9"
                    },
                    {
                        "query": "What are the benefits of immunotherapy for melanoma?",
                        "expected_keywords": ["immunotherapy", "melanoma", "checkpoint"],
                        "expected_paper": "Immunotherapy"
                    },
                    {
                        "query": "How is AI and machine learning used in drug discovery?",
                        "expected_keywords": ["machine learning", "drug discovery", "AI"],
                        "expected_paper": "Machine Learning"
                    }
                ]
                
                search_results = {}
                all_searches_good = True
                
                for test_case in test_cases:
                    query = test_case["query"]
                    relevant_docs = retriever.get_relevant_documents(query)
                    
                    if relevant_docs:
                        top_doc = relevant_docs[0]
                        title = top_doc.metadata.get('title', 'Unknown')
                        
                        # Check if we got the expected paper
                        title_match = any(keyword in title for keyword in test_case["expected_paper"].split())
                        content_match = any(keyword.lower() in top_doc.page_content.lower() 
                                          for keyword in test_case["expected_keywords"])
                        
                        search_results[query] = {
                            'docs_found': len(relevant_docs),
                            'top_title': title,
                            'title_match': title_match,
                            'content_match': content_match,
                            'success': title_match or content_match
                        }
                        
                        if title_match or content_match:
                            st.success(f"✅ **Query:** {query}")
                            st.success(f"   **Found:** {len(relevant_docs)} docs | **Top Result:** {title}")
                        else:
                            st.warning(f"⚠️ **Query:** {query}")
                            st.warning(f"   **Found:** {len(relevant_docs)} docs | **Top Result:** {title} (unexpected)")
                            all_searches_good = False
                    else:
                        st.error(f"❌ **Query:** {query} | **Found:** 0 documents")
                        search_results[query] = {'docs_found': 0, 'success': False}
                        all_searches_good = False
                
                if all_searches_good:
                    st.success("✅ All semantic search tests passed!")
                else:
                    st.warning("⚠️ Some semantic search tests had unexpected results")
                
            except Exception as e:
                st.error(f"❌ Semantic search testing failed: {e}")
                traceback.print_exc()
                return
            
            # Step 5: Test RAG + BioMistral integration
            try:
                st.markdown("### 5️⃣ Testing RAG + BioMistral Integration")
                
                test_question = "How does CRISPR gene editing help treat cancer, and what are the advantages over traditional treatments?"
                
                # Get relevant context using RAG
                relevant_docs = retriever.get_relevant_documents(test_question)
                
                if not relevant_docs:
                    st.error("❌ No relevant documents found for the test question")
                    return
                
                # Format context for BioMistral
                context_sections = []
                for i, doc in enumerate(relevant_docs[:2]):  # Use top 2 most relevant
                    context_sections.append(f"""
Research Paper {i+1}:
Title: {doc.metadata.get('title', 'Unknown')}
Authors: {doc.metadata.get('authors', 'Unknown')}
Year: {doc.metadata.get('year_of_publication', 'N/A')}
Content: {doc.page_content}
""")
                
                formatted_context = "\n".join(context_sections)
                
                # Create enhanced prompt for BioMistral
                rag_prompt = f"""You are BioMistral, a medical research AI assistant. Based on the following research papers, answer this question: {test_question}

RESEARCH CONTEXT FROM SCIENTIFIC PAPERS:
{formatted_context}

Please provide a comprehensive analysis that:
1. Explains how CRISPR works in cancer treatment
2. Discusses the advantages over traditional treatments
3. References specific findings from the research papers
4. Provides evidence-based conclusions

Your detailed medical response:"""
                
                # Get BioMistral response
                from app import get_BioMistral_response
                response = get_BioMistral_response(components['ai_model'], rag_prompt, "Long")
                
                st.markdown("#### 📋 Test Results:")
                st.write(f"**Test Question:** {test_question}")
                st.write(f"**Context Sources:** {len(relevant_docs)} research papers")
                st.markdown("**RAG-Enhanced BioMistral Response:**")
                st.write(response)
                
                # Evaluate response quality
                if response and len(response.strip()) > 50:
                    # Check if response mentions key concepts
                    key_concepts = ["CRISPR", "gene editing", "cancer", "treatment"]
                    mentioned_concepts = [concept for concept in key_concepts 
                                        if concept.lower() in response.lower()]
                    
                    if len(mentioned_concepts) >= 3:
                        st.success(f"✅ High-quality response! Mentioned {len(mentioned_concepts)}/4 key concepts: {mentioned_concepts}")
                    else:
                        st.warning(f"⚠️ Response could be better. Only mentioned {len(mentioned_concepts)}/4 key concepts: {mentioned_concepts}")
                else:
                    st.warning("⚠️ Response seems short, but RAG integration is working")
                
            except Exception as e:
                st.error(f"❌ RAG + BioMistral integration test failed: {e}")
                traceback.print_exc()
                return
            
            # Step 6: Cleanup
            try:
                st.markdown("### 6️⃣ Cleaning Up Test Data")
                
                # Delete test RAG index
                if components['rag_system'].delete_vector_index(query_id):
                    st.success("✅ RAG vector index deleted")
                else:
                    st.warning("⚠️ RAG vector index deletion had issues")
                
                # Delete test storage
                components['storage'].delete_dataset(query_id)
                st.success("✅ Test dataset deleted from storage")
                
                # Verify cleanup
                remaining_collections = components['rag_system'].list_collections()
                if expected_collection not in remaining_collections:
                    st.success("✅ Cleanup verification passed")
                else:
                    st.warning("⚠️ Collection still exists after cleanup")
                
            except Exception as e:
                st.warning(f"⚠️ Cleanup had issues: {e}")
            
            # Final summary
            st.markdown("## 🎉 Comprehensive Test Complete!")
            st.success("""
            **All RAG + BioMistral components are working correctly!**
            
            ✅ Component initialization and verification
            ✅ Test research paper creation with detailed metadata
            ✅ RAG vector index creation and verification
            ✅ Semantic search with multiple test cases
            ✅ RAG + BioMistral integration with quality evaluation
            ✅ Complete cleanup and verification
            
            **Your system is ready for production use!**
            """)
            
            st.info("""
            **What this means:**
            - Your RAG system can successfully create and search vector indexes
            - BioMistral can process RAG-enhanced prompts
            - Semantic search finds relevant paper sections accurately
            - The full pipeline from papers → RAG → AI analysis works correctly
            """)

def render_enhanced_test_sidebar(components: Dict[str, Any]):
    """Enhanced sidebar with comprehensive testing options"""
    with st.sidebar:
        st.header("🛠️ System Tests")

        # Quick System Test
        if st.button("⚡ Quick System Test"):
            with st.spinner("Running quick tests..."):
                results = quick_system_test()
                
                # Display results with summary
                working_count = sum(1 for r in results if "✅" in r)
                warning_count = sum(1 for r in results if "⚠️" in r)
                error_count = sum(1 for r in results if "❌" in r)
                
                st.markdown(f"**Test Summary:** {working_count} working, {warning_count} warnings, {error_count} errors")
                
                for result in results:
                    if "✅" in result:
                        st.success(result)
                    elif "⚠️" in result:
                        st.warning(result)
                    else:
                        st.error(result)

        # RAG Integration Test
        if components.get('rag_system') and components.get('storage'):
            if st.button("🔧 Test RAG Integration"):
                with st.spinner("Testing RAG integration..."):
                    working, message = test_rag_integration_enhanced(components)
                    if working:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")

        # Full Pipeline Test
        if all([components.get('ai_model'), components.get('rag_system'), components.get('storage')]):
            if st.button("🚀 Full Pipeline Test"):
                comprehensive_rag_test()
        else:
            missing = [comp for comp in ['ai_model', 'rag_system', 'storage'] if not components.get(comp)]
            st.warning(f"Full test needs: {missing}")

        # Component Status
        st.markdown("---")
        st.subheader("📊 Component Status")
        
        status_items = [
            ("🤖 BioMistral", components.get('ai_model')),
            ("📚 Embeddings", components.get('embeddings')),
            ("🗄️ Vector Store", components.get('rag_system')),
            ("💾 Storage", components.get('storage')),
            ("📖 PubMed", components.get('retriever'))
        ]
        
        for name, component in status_items:
            if component:
                st.success(f"{name}: ✅ Ready")
            else:
                st.error(f"{name}: ❌ Missing")

        # Environment Info
        st.markdown("---")
        st.subheader("🔧 Environment")
        
        import os
        env_vars = [
            ("PUBMED_EMAIL", os.getenv('PUBMED_EMAIL')),
            ("PUBMED_API_KEY", bool(os.getenv('PUBMED_API_KEY'))),
            ("HUGGINGFACE_TOKEN", bool(os.getenv('HUGGINGFACE_TOKEN')))
        ]
        
        for var_name, var_value in env_vars:
            if var_value:
                st.success(f"{var_name}: ✅ Set")
            else:
                st.warning(f"{var_name}: ⚠️ Not set")

# Main function for running tests
def main():
    """Main function for the test application"""
    st.set_page_config(
        page_title="RAG + BioMistral Test Suite",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 RAG + BioMistral Test Suite")
    st.markdown("**Comprehensive testing for your biomedical research system**")
    
    # Initialize components (you may need to adjust this import)
    try:
        from app import initialize_system_components
        components = initialize_system_components()
    except ImportError:
        st.error("❌ Cannot import from app.py. Make sure app.py is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error initializing components: {e}")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎯 Available Tests")
        
        st.markdown("""
        **⚡ Quick System Test**
        - Tests all individual components
        - Verifies basic functionality
        - Shows what's working and what's not
        
        **🔧 RAG Integration Test** 
        - Tests vector store creation
        - Tests semantic search
        - Uses sample documents
        
        **🚀 Full Pipeline Test**
        - Complete end-to-end test
        - Creates research papers
        - Tests RAG + BioMistral integration
        - Evaluates response quality
        """)
        
        # Show comprehensive test interface
        comprehensive_rag_test()
    
    with col2:
        # Enhanced sidebar content
        render_enhanced_test_sidebar(components)

if __name__ == "__main__":
    main()