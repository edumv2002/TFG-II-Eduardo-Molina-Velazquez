package fairness;

import db.DecideUSADBManager;
import entities.DUSAProposal;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class DUSAProposalAnnotator {

    private static String indexPath;

    private static Map<String, String> ambiguities;

    static {
        ambiguities = new HashMap<>(); 
    }

    public static void createIndex(String database) throws Exception {
        System.out.println("Indexing to directory '" + indexPath + "'...");

        Directory dir = FSDirectory.open(Paths.get(indexPath));
        Analyzer analyzer = new StandardAnalyzer();
        IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
        iwc.setOpenMode(OpenMode.CREATE);
        IndexWriter writer = new IndexWriter(dir, iwc);
    
        DecideUSADBManager db = new DecideUSADBManager();
        List<DUSAProposal> proposals = db.selectProposals(database);
        for (DUSAProposal p : proposals) {
            int id = p.getId();
            String dataset = p.getDataset();
            String title = p.getTitle() != null ? p.getTitle() : "";
            String title_gpt = p.getTitle_gpt() != null ? p.getTitle_gpt() : "";
            String description = p.getDescription() != null ? p.getDescription() : "";
            String category = p.getCategory() != null ? p.getCategory() : "";
            String timestamp = p.getTimestamp() != null ? p.getTimestamp() : "";
            float longitude = p.getLongitude();
            float latitude = p.getLatitude();

            for (String key : ambiguities.keySet()) {
                String value = ambiguities.get(key);

                title = title.replace(key, value);
                title_gpt = title_gpt.replace(key, value);
                description = description.replace(key, value);
                category = category.replace(key, value);
            }

            Document doc = new Document();
            doc.add(new StringField("id", "" + id, Field.Store.YES));
            doc.add(new StringField("dataset", "" + dataset, Field.Store.YES));
            doc.add(new TextField("title", title, Field.Store.YES));
            doc.add(new TextField("title_gpt", title_gpt, Field.Store.YES));
            doc.add(new TextField("description", description == null ? "" : description, Field.Store.YES));
            doc.add(new TextField("category", category == null ? "" : category, Field.Store.YES));
            doc.add(new StringField("timestamp", timestamp == null ? "" : timestamp, Field.Store.YES));
            doc.add(new StringField("longitude", "" + longitude, Field.Store.YES));
            doc.add(new StringField("latitude", "" + latitude, Field.Store.YES));
            writer.addDocument(doc);
        }
        writer.close();
    }

    private static Map<String, List<String>> readGroupVocabularies(String filename) throws Exception {
        Map<String, List<String>> groupVocabularies = new HashMap<>();

        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), StandardCharsets.UTF_8));
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.trim().isEmpty()) {
                continue;
            }
            if (line.startsWith("#")) {
                break;
            }
            String[] tokens = line.split("\t");
            String groupName = tokens[1];  // example: nimby    Black People    black, racist, racism
            List<String> groupKeywords = new ArrayList<>();
            StringTokenizer tokenizer = new StringTokenizer(tokens[2], ",");
            while (tokenizer.hasMoreTokens()) {
                String keyword = tokenizer.nextToken().toLowerCase();
                groupKeywords.add(keyword);
            }

            groupVocabularies.put(groupName, groupKeywords);
        }
        reader.close();

        return groupVocabularies;
    }

    private static Map<String, String> readGroupTypes(String file) throws Exception {
        Map<String, String> groupTypes = new HashMap<>();

        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8));

        String line = null;
        while ((line = reader.readLine()) != null) {
            if (line.trim().isEmpty()) {
                continue;
            }
            if (line.startsWith("#")) {
                break;
            }
            String[] tokens = line.split("\t");
            String groupType = tokens[0];
            String groupName = tokens[1];
            groupTypes.put(groupName, groupType);
        }

        reader.close();
        
        return groupTypes;
    }

    private static void search(Map<String, String> groupTypes, Map<String, List<String>> groupVocabularies, Map<String, Float> searchFields, String outputFile) throws Exception {
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), StandardCharsets.UTF_8));
        writer.write("type|group|ranking|id|score|title|title_gpt|description|category|timestamp|longitude|latitude\n");

        for (String groupName : groupVocabularies.keySet()) {
            String groupType = groupTypes.get(groupName);
            List<String> groupKeywords = groupVocabularies.get(groupName);
            DUSAProposalAnnotator.search(groupName, groupType, groupKeywords, searchFields, writer);
        }
        writer.close();
    }

    private static void search(String group, String type, List<String> keywords, Map<String, Float> searchFields, BufferedWriter writer) throws Exception {
        String queryString = buildQueryString(keywords);
        
        DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexPath)));
        IndexSearcher searcher = new IndexSearcher(reader);
        Analyzer analyzer = new StandardAnalyzer();
        StoredFields storedFields = searcher.storedFields();

        BooleanQuery.Builder booleanQueryBuilder = new BooleanQuery.Builder();
        for (String field : searchFields.keySet()) {
            float weight = searchFields.get(field);
            
            booleanQueryBuilder.add(new BoostQuery(new QueryParser(field, analyzer).parse(queryString), weight), BooleanClause.Occur.SHOULD);

        }
        Query booleanQuery = booleanQueryBuilder.build();

        System.out.println("Query: " + booleanQuery.toString());
        TopDocs results = searcher.search(booleanQuery, 10000);
        ScoreDoc[] hits = results.scoreDocs;

        System.out.println("Number of results: " + hits.length);
        for (int i = 0; i < hits.length; i++) {
            Document doc = searcher.doc(hits[i].doc);
            int id = Integer.valueOf(doc.get("id"));
            float score = hits[i].score;

            String dataset = doc.get("dataset");
            String title = doc.get("title").replace(FIELD_SEPARATOR, " ");
            String title_gpt = doc.get("title_gpt").replace(FIELD_SEPARATOR, " ");
            String description = doc.get("description").replace(FIELD_SEPARATOR, " ");
            String category = doc.get("category").replace(FIELD_SEPARATOR, " ");
            String timestamp = doc.get("timestamp").replace(FIELD_SEPARATOR, " ");
            float longitude = Float.valueOf(doc.get("longitude"));
            float latitude = Float.valueOf(doc.get("latitude"));

            if (title.isEmpty()) {
                title = " ";
            }
            if (title_gpt.isEmpty()) {
                title_gpt = " ";
            }
            if (description.isEmpty()) {
                description = " ";
            }
            if (category.isEmpty()) {
                category = " ";
            }
            if (timestamp.isEmpty()) {
                timestamp = " ";
            }
            writer.write(type + "|" + group + "|" + (i + 1) + "|" + id + "|" + (("" + score).replace(".", DECIMAL_SEPARATOR)) + "|" + title + "|" + title_gpt + "|" + description + "|" + category + "|" + timestamp + "|" + (("" + longitude).replace(".", DECIMAL_SEPARATOR)) + "|" + (("" + latitude).replace(".", DECIMAL_SEPARATOR)) + "\n");

            System.out.println("\t" + id + " (" + (("" + score).replace(".", DECIMAL_SEPARATOR)) + "): " + title_gpt);
        }
    }

    private static String buildQueryString(List<String> keywords) {
        String queryString = "";

        for (int k = 0; k < keywords.size(); k++) {
            String keyword = keywords.get(k);
            queryString += " " + (keyword.contains(" ") ? "\"" + keyword + "\"" : keyword);

            String keyword2 = keyword.toLowerCase()
                    .replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
                    .replace("à", "a").replace("è", "e").replace("ì", "i").replace("ò", "o").replace("ù", "u")
                    .replace("ä", "a").replace("ë", "e").replace("ï", "i").replace("ö", "o").replace("ü", "u")
                    .replace("â", "a").replace("ê", "e").replace("î", "i").replace("ô", "o").replace("û", "u")
                    .replace("ç", "c");
            if (!keyword.equals(keyword2)) {
                queryString += " " + (keyword2.contains(" ") ? "\"" + keyword2 + "\"" : keyword2);
            }
        }

        queryString = queryString.trim();

        return queryString;
    }
    
    public static final String DECIMAL_SEPARATOR = ",";
    public static final String FIELD_SEPARATOR = "|";

    public static void main(String[] args) {
        try {
            String city = args[0];
            String year = args[1];
            String dataset = args[2];
            // indexPath = "./data/indexes2/indexMiami2014/";
            indexPath = args[3];
            DUSAProposalAnnotator.createIndex(dataset);
            Map<String, String> groupTypes = DUSAProposalAnnotator.readGroupTypes("./data/results2/"+city+"/vocabulary/"+city+"_"+year+"_vocabulary.txt");
            Map<String, List<String>> groupVocabularies = DUSAProposalAnnotator.readGroupVocabularies("./data/results2/"+city+"/vocabulary/"+city+"_"+year+"_vocabulary.txt");

            Map<String, Float> searchFields = new HashMap<>();
            searchFields.put("title_gpt", 100.0f);
            DUSAProposalAnnotator.search(groupTypes, groupVocabularies, searchFields, "./data/results2/"+city+"/"+year+"/prop_gpt.csv");
            searchFields.put("description", 10.0f);
            DUSAProposalAnnotator.search(groupTypes, groupVocabularies, searchFields, "./data/results2/"+city+"/"+year+"/prop_gpt_desc.csv");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}