package db;

import entities.DUSAProposal;
import entities.DUSATopic;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.List;

public class DecideUSADBManager {

    private MySQLDBConnector db;

    public DecideUSADBManager() throws Exception {
        this.db = new MySQLDBConnector();
        this.db.connect("localhost", "participatory_budgeting", "eduardomv", "*****");
    }

    @Override
    public void finalize() {
        this.db.disconnect();
    }

    public List<DUSAProposal> selectProposals(String database) throws Exception {
        List<DUSAProposal> proposals = new ArrayList<>();

        String query = "SELECT * FROM items where dataset = '" + database + "'";
        ResultSet rs = this.db.executeSelect(query);
        while (rs.next()) {
            int id = rs.getInt("id");
            String dataset = rs.getString("dataset");
            String title = rs.getString("title");
            String title_gpt = rs.getString("title_gpt");
            String description = rs.getString("description");
            String category = rs.getString("category");
            String timestamp = rs.getString("timestamp");
            float longitude = rs.getFloat("longitude");
            float latitude = rs.getFloat("latitude");

            DUSAProposal proposal = new DUSAProposal(id, dataset, title, title_gpt, description, category, timestamp, longitude, latitude);
            proposals.add(proposal);
        }
        rs.close();

        return proposals;
    }

    public List<DUSATopic> selectTopics(String database) throws Exception {
        List <DUSATopic> topics = new ArrayList<>();
        String query = String.format("""
        SELECT it.dataset, it.id, it.title_gpt, sc.topicLabel 
        FROM items it 
        LEFT JOIN semantic_topics sc ON it.id = sc.id
        WHERE it.dataset='%s'
        """, database);
        ResultSet rs = this.db.executeSelect(query);
        while (rs.next()) {
            String dataset = rs.getString("dataset");
            int itemId = rs.getInt("id");
            String title_gpt = rs.getString("title_gpt");
            String topicLabel = rs.getString("topicLabel");

            DUSATopic topic = new DUSATopic(dataset, itemId, title_gpt, topicLabel);
            topics.add(topic);
        }
        rs.close();

        return topics;
    }

}
