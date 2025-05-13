package entities;

public class DUSAProposal {
    private int id;
    private String dataset;
    private String title;
    private String title_gpt;
    private String description;
    private String category;
    private String timestamp;
    private float longitude;
    private float latitude;

    public DUSAProposal(int id, String dataset, String title, String title_gpt, String description, String category, String timestamp, float longitude, float latitude) {
        this.id = id;
        this.dataset = dataset;
        this.title = title;
        this.title_gpt = title_gpt;
        this.description = description.trim().isEmpty() ? null : description.trim();
        this.category = category.trim().isEmpty() ? null : category.trim();
        this.timestamp = timestamp.trim().isEmpty() ? null : timestamp.trim();
        this.longitude = longitude;
        this.latitude = latitude;
    }

    public int getId() {
        return this.id;
    }

    public String getDataset() {
        return this.dataset;
    }

    public String getTitle() {
        return this.title;
    }

    public String getTitle_gpt() {
        return this.title_gpt;
    }

    public String getDescription() {
        return this.description;
    }

    public String getCategory() {
        return this.category;
    }

    public String getTimestamp() {
        return this.timestamp;
    }

    public float getLongitude() {
        return this.longitude;
    }

    public float getLatitude() {
        return this.latitude;
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 37 * hash + this.id;
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final DUSAProposal other = (DUSAProposal) obj;
        if (this.id != other.id) {
            return false;
        }
        return true;
    }

    @Override
    public String toString() {
        return "DUSAProposal{" + "id=" + this.id + ", dataset=" + this.dataset + ", title=" + this.title + ", title_gpt=" + this.title_gpt + ", description=" + this.description + ", category=" + this.category + ", timestamp=" + this.timestamp + ", longitude=" + this.longitude + ", latitude=" + this.latitude + '}';
    }
}